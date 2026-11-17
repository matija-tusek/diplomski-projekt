import argparse
import json
from collections import Counter
from pathlib import Path

import esco_loader
from esco_embeddings import VARIANTS, embedding_lookup, load_embeddings
from esco_loader import load_esco_index, map_term, remove_seniority

KINDS = ["skills", "occupations"]


def check(result, expected):
    # expected = popis ESCO naziva koje prihvacamo, prazan popis znaci da treba ostati UNMAPPED
    concept = result["esco_concept"]
    if concept is None:
        return "correct" if not expected else "missed"
    if concept["preferred_label"] in expected:
        return "correct"
    return "wrong"


def run(gold, data):
    rows = []
    for kind in KINDS:
        index, embeddings = data[kind]
        for item in gold[kind]:
            result = map_term(item["term"], index, kind == "occupations", embeddings)
            rows.append({
                "kind": kind,
                "term": item["term"],
                "method": result["mapping"]["method"],
                "found": (result["esco_concept"] or {}).get("preferred_label"),
                "expected": item["expected"],
                "verdict": check(result, item["expected"]),
            })
    return rows


def summary(rows):
    mapped = [r for r in rows if r["method"] != "none"]
    mapped_ok = [r for r in mapped if r["verdict"] == "correct"]
    all_ok = [r for r in rows if r["verdict"] == "correct"]
    precision = len(mapped_ok) / len(mapped) if mapped else 0
    return (f"mapped {len(mapped)}/{len(rows)}, correct when mapped {len(mapped_ok)}/{len(mapped)} "
            f"(precision {precision:.0%}), overall correct {len(all_ok)}/{len(rows)} ({len(all_ok) / len(rows):.0%})")


def print_report(rows):
    for kind in KINDS + ["all"]:
        part = [r for r in rows if kind == "all" or r["kind"] == kind]
        print(f"  {kind:12} {summary(part)}")

    print("\n  method                 count  correct  wrong  missed")
    for method in ["exact_lookup", "alias_lookup", "fuzzy_match", "embedding_similarity", "none"]:
        verdicts = Counter(r["verdict"] for r in rows if r["method"] == method)
        count = sum(verdicts.values())
        print(f"  {method:22} {count:5}  {verdicts['correct']:7}  {verdicts['wrong']:5}  {verdicts['missed']:6}")

    print("\n  Errors:")
    for r in rows:
        if r["verdict"] == "wrong":
            print(f"    WRONG  {r['term']!r} -> {r['found']!r} ({r['method']}), expected {r['expected']}")
    for r in rows:
        if r["verdict"] == "missed":
            print(f"    MISSED {r['term']!r} -> UNMAPPED, expected {r['expected']}")


def recall_at_k(gold, data, k=5):
    # samo embedding, bez exact i fuzzy: je li tocan odgovor prvi, odnosno barem u top-k?
    # (koristi se za odluku isplati li se rerank top-k kandidata)
    for kind in KINDS:
        embeddings = data[kind][1]
        items = [item for item in gold[kind] if item["expected"]]
        top1 = topk = 0
        for item in items:
            query = remove_seniority(item["term"]) if kind == "occupations" else item["term"]
            labels = [concept["preferred_label"] for concept, _, _ in embedding_lookup(query, embeddings, k)]
            top1 += labels[0] in item["expected"]
            topk += any(label in item["expected"] for label in labels)
        print(f"  {kind:12} recall@1 {top1}/{len(items)} ({top1 / len(items):.0%}), "
              f"recall@{k} {topk}/{len(items)} ({topk / len(items):.0%})")


def check_gold_labels(gold, data):
    # da nam se ne potkrade tipfeler u gold setu
    for kind in KINDS:
        index = data[kind][0]
        known = {entry[0]["preferred_label"] for entries in index.values() for entry in entries}
        for item in gold[kind]:
            for label in item["expected"]:
                if label not in known:
                    print(f"  WARNING: {label!r} (gold for {item['term']!r}) is not an ESCO {kind} label")


def coverage(terms_path, data):
    # za prave podatke bez gold oznaka: samo koliko se pojmova mapira kojom metodom
    terms = [line.strip() for line in Path(terms_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    index, embeddings = data["skills"]
    methods = Counter(map_term(term, index, embeddings=embeddings)["mapping"]["method"] for term in terms)
    print(f"\n=== Coverage on {terms_path} ({len(terms)} terms)")
    for method, count in methods.most_common():
        print(f"  {method:22} {count:5}  ({count / len(terms):.0%})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESCO normalization on the gold set")
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--occupations", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=Path(__file__).parent / "eval" / "gold_terms.json")
    parser.add_argument("--terms", type=Path, help="optional .txt with one skill per line (e.g. from a real dataset)")
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    data = {
        "skills": (load_esco_index(args.skills), load_embeddings(args.skills)),
        "occupations": (load_esco_index(args.occupations), load_embeddings(args.occupations)),
    }
    check_gold_labels(gold, data)

    print(f"\n=== Default settings (fuzzy {esco_loader.FUZZY_CUTOFF}/{esco_loader.FUZZY_CUTOFF_OCCUPATION}, "
          f"embedding {esco_loader.EMBEDDING_CUTOFF})")
    print_report(run(gold, data))

    print("\n=== Embedding cutoff")
    default_cutoff = esco_loader.EMBEDDING_CUTOFF
    for cutoff in [0.70, 0.75, 0.78, 0.80, 0.85, 0.90]:
        esco_loader.EMBEDDING_CUTOFF = cutoff
        print(f"  {cutoff:.2f}  {summary(run(gold, data))}")
    esco_loader.EMBEDDING_CUTOFF = default_cutoff

    print("\n=== Fuzzy cutoff (skills)")
    default_cutoff = esco_loader.FUZZY_CUTOFF
    for cutoff in [80, 83, 85, 88, 90]:
        esco_loader.FUZZY_CUTOFF = cutoff
        print(f"  {cutoff}  {summary(run(gold, data))}")
    esco_loader.FUZZY_CUTOFF = default_cutoff

    # sto embedati za svaki ESCO koncept (labels je ono sto sustav koristi)
    for variant in VARIANTS:
        variant_data = data
        if variant != "labels":
            variant_data = {
                "skills": (data["skills"][0], load_embeddings(args.skills, variant)),
                "occupations": (data["occupations"][0], load_embeddings(args.occupations, variant)),
            }
        print(f"\n=== Embedding variant '{variant}'")
        print(f"  all          {summary(run(gold, variant_data))}")
        recall_at_k(gold, variant_data)

    if args.terms:
        coverage(args.terms, data)


if __name__ == "__main__":
    main()
