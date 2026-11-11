"""Command line: extract CandidateProfiles / JobProfiles from a file, folder or dataset.

Examples (run from the repo root):
    python -m extraction.cli cv  extraction/samples/cv.txt
    python -m extraction.cli job extraction/samples/job.txt
    python -m extraction.cli job data/jobs.csv --text-column description --id-column id --limit 20

One document  -> output/<id>.json
Many documents -> output/<cv|job>_profiles.jsonl (one profile per line)
Every run also prints evidence statistics and the items dropped for ungrounded evidence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .datasets import load_documents
from .extractor import MODEL, extract_candidate, extract_job


def run(kind, doc_id, text, dataset):
    if kind == "cv":
        return extract_candidate(text, f"cand_{doc_id}")
    return extract_job(text, f"job_{doc_id}", dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Group 1: LLM extraction to schema v1.0")
    parser.add_argument("kind", choices=["cv", "job"])
    parser.add_argument("input", type=Path, help=".txt file, folder of .txt, or .csv/.json/.jsonl")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column")
    parser.add_argument("--dataset", help="dataset name stored in JobProfile.source (default: input file name)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, help="output file (default under output/)")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    dataset = args.dataset or args.input.stem
    docs = list(load_documents(args.input, args.text_column, args.id_column, args.limit))
    if not docs:
        sys.exit("No documents found.")
    print(f"Model {MODEL}: extracting {len(docs)} {args.kind} document(s)")

    profiles, total_found, total_dropped = [], 0, 0
    for doc_id, text in docs:
        t0 = time.perf_counter()
        profile, report = run(args.kind, doc_id, text, dataset)
        profiles.append(profile)
        total_found += report["evidence_found"]
        total_dropped += len(report["dropped"])
        print(f"- {doc_id}: {time.perf_counter() - t0:.1f}s, evidence found "
              f"{report['evidence_found']}, dropped {len(report['dropped'])}")
        for d in report["dropped"]:
            print(f"    dropped {d['item']!r}: quote not in source: {d['quote']!r}")

    if len(profiles) == 1:
        out = args.out or Path("output") / f"{profiles[0].get('candidate_id') or profiles[0]['job_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profiles[0], indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        out = args.out or Path("output") / f"{args.kind}_profiles.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for p in profiles:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    grounded = total_found / (total_found + total_dropped) if total_found + total_dropped else 1.0
    print(f"Evidence grounded: {total_found}/{total_found + total_dropped} ({grounded:.0%}). Saved to {out}")


if __name__ == "__main__":
    main()
