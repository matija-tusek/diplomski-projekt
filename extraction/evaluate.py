"""Measure extraction quality on the hand-labelled gold set (extraction/gold/).

    python -m extraction.evaluate            # run the LLM on every gold document, then score
    python -m extraction.evaluate --reuse    # score the cached predictions in output/gold/ again

Metrics:
- list fields (skills, languages, education, experience): precision / recall / F1.
  Items are matched by name; gold alternatives "a|b" and lowercased letters/digits are used.
- attributes of matched items (modality, CEFR level, months, field, status): accuracy.
- single values (is_student, job_type, remote_mode, city, ...): accuracy, where null must
  be predicted as null ("unknown is not false").
- evidence: share of LLM quotes found in the source (items with ungrounded quotes are dropped).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .extractor import MODEL, extract_candidate, extract_job

GOLD = Path(__file__).parent / "gold"
CACHE = Path("output") / "gold"


def norm(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def alts(gold):
    return {norm(a) for a in str(gold).split("|")}


def same(gold, pred):
    """Exact match on normalized text, allowing gold alternatives; None only equals None."""
    if gold is None or pred is None:
        return gold is None and pred is None
    return norm(pred) in alts(gold)


def contains(gold, pred):
    """For free-text fields: a gold alternative appears inside the prediction or vice versa."""
    if gold is None or pred is None:
        return gold is None and pred is None
    p = norm(pred)
    return any(a and (a in p or p in a) for a in alts(gold)) if p else False


class Scores:
    def __init__(self):
        self.prf = defaultdict(lambda: [0, 0, 0])   # field -> [true positives, predicted, gold]
        self.acc = defaultdict(lambda: [0, 0])      # field -> [correct, total]
        self.errors = []

    def match_list(self, field, doc, gold_items, pred_items, key, ignore=(), matcher=same):
        """Greedy one-to-one matching. Returns [(gold, pred)] pairs."""
        unused = list(pred_items)
        pairs = []
        for g in gold_items:
            hit = next((p for p in unused if matcher(key(g), key(p))), None)
            if hit is None:
                self.errors.append(f"{doc} {field}: missed {key(g)!r}")
            else:
                unused.remove(hit)
                pairs.append((g, hit))
        extra = [p for p in unused if not any(same(i, key(p)) for i in ignore)]
        for p in extra:
            self.errors.append(f"{doc} {field}: extra {key(p)!r}")
        t = self.prf[field]
        t[0] += len(pairs)
        t[1] += len(pairs) + len(extra)
        t[2] += len(gold_items)
        return pairs

    def check(self, field, doc, gold, pred, compare=same):
        ok = compare(gold, pred)
        self.acc[field][0] += ok
        self.acc[field][1] += 1
        if not ok:
            self.errors.append(f"{doc} {field}: expected {gold!r}, got {pred!r}")


def score_cv(s, doc, gold, pred):
    s.check("cv.is_student", doc, gold["is_student"], pred["student_status"]["is_student"])
    for g, p in s.match_list("cv.education", doc, gold["education"], pred["education"], lambda x: x["level"]):
        s.check("cv.education.status", doc, g["status"], p["status"])
        s.check("cv.education.field", doc, g["field"], p["field"], contains)
    s.match_list("cv.skills", doc, gold["skills"], pred["skills"], lambda x: x if isinstance(x, str) else x["name"],
                 ignore=gold["skills_optional"])
    for g, p in s.match_list("cv.languages", doc, gold["languages"], pred["languages"], lambda x: x["language"]):
        s.check("cv.languages.level", doc, g["level"], p["level"])
    for g, p in s.match_list("cv.experience", doc, gold["experience"], pred["experience"], lambda x: x["title"],
                             matcher=contains):
        s.check("cv.experience.months", doc, g["months"], p["months"])
    s.check("cv.preferences.remote_mode", doc, gold["remote_mode"], pred["preferences"]["remote_mode"])
    s.check("cv.preferences.job_types", doc, sorted(gold["job_types"] or []),
            sorted(pred["preferences"]["job_types"] or []), lambda a, b: a == b)


def score_job(s, doc, gold, pred):
    s.check("job.job_type", doc, gold["job_type"], pred["job_type"])
    s.check("job.remote_mode", doc, gold["remote_mode"], (pred["remote_mode"] or {}).get("value"))
    s.check("job.location.city", doc, gold["city"], pred["location"]["city"])
    s.check("job.location.country", doc, gold["country"], pred["location"]["country"])
    for g, p in s.match_list("job.skills", doc, gold["skills"], pred["skills"],
                             lambda x: x if isinstance(x, str) else x["name"], ignore=gold["skills_optional"]):
        s.check("job.skills.modality", doc, g["modality"], p["modality"])
    for g, p in s.match_list("job.languages", doc, gold["languages"], pred["languages"], lambda x: x["language"]):
        s.check("job.languages.level", doc, g["level"], p["level"])
        s.check("job.languages.modality", doc, g["modality"], p["modality"])
    edu = pred["education"]
    s.check("job.education.level", doc, gold["education"] and gold["education"]["level"], edu and edu["level"])
    s.check("job.student_status_required", doc, gold["student_status_required"],
            (pred["student_status_required"] or {}).get("value"))
    exp = pred["experience"]
    s.check("job.experience.min_years", doc, gold["experience_min_years"], exp and exp["min_years"],
            lambda a, b: a == b)


def predict(kind, doc, reuse):
    cached = CACHE / f"{doc}.json"
    if reuse and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    text = (GOLD / kind / f"{doc}.txt").read_text(encoding="utf-8")
    if kind == "cv":
        profile, report = extract_candidate(text, doc.replace("cv_", "cand_"))
    else:
        profile, report = extract_job(text, doc, "gold")
    result = {"profile": profile, "report": report}
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score extraction on the gold set")
    parser.add_argument("--reuse", action="store_true", help="use cached predictions from output/gold/")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    labels = json.loads((GOLD / "labels.json").read_text(encoding="utf-8"))
    s = Scores()
    found = dropped = 0
    for kind, scorer in (("cv", score_cv), ("job", score_job)):
        for doc, gold in labels[kind].items():
            print(f"{doc} ...", flush=True)
            result = predict(kind, doc, args.reuse)
            found += result["report"]["evidence_found"]
            dropped += len(result["report"]["dropped"])
            scorer(s, doc, gold, result["profile"])

    lines = [f"Model: {MODEL}", "", f"{'list field':32} {'P':>5} {'R':>5} {'F1':>5}   (tp/pred/gold)"]
    summary = {"model": MODEL, "lists": {}, "accuracy": {}}
    for field, (tp, n_pred, n_gold) in s.prf.items():
        p = tp / n_pred if n_pred else 1.0
        r = tp / n_gold if n_gold else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        summary["lists"][field] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred": n_pred, "gold": n_gold}
        lines.append(f"{field:32} {p:5.2f} {r:5.2f} {f1:5.2f}   ({tp}/{n_pred}/{n_gold})")
    lines += ["", f"{'value':32} {'acc':>5}   (correct/total)"]
    for field, (ok, n) in s.acc.items():
        summary["accuracy"][field] = {"accuracy": ok / n, "correct": ok, "total": n}
        lines.append(f"{field:32} {ok / n:5.2f}   ({ok}/{n})")
    grounded = found / (found + dropped) if found + dropped else 1.0
    summary["evidence"] = {"found": found, "dropped": dropped, "grounded_rate": grounded}
    summary["errors"] = s.errors
    lines += ["", f"Evidence quotes grounded: {found}/{found + dropped} ({grounded:.0%})",
              "", f"Errors ({len(s.errors)}):"] + [f"  {e}" for e in s.errors]
    print("\n".join(lines))

    out = Path("output") / "eval_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
