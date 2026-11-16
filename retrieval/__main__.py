"""Pokretanje iz komandne linije.

  python -m retrieval build --jobs retrieval/examples/jobs.json --index faiss_index
  python -m retrieval query --candidates retrieval/examples/candidates.json --index faiss_index --k 50
"""

import argparse
import json

from .index import DEFAULT_ENCODER, JobIndex
from .io_utils import load_profiles, save_json
from .retrieve import retrieve


def main():
    parser = argparse.ArgumentParser(prog="python -m retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="enkodira oglase i sprema FAISS indeks")
    b.add_argument("--jobs", required=True, help="NormalizedProfile oglasa (.json, .jsonl ili mapa)")
    b.add_argument("--index", default="faiss_index", help="mapa u koju se sprema indeks")
    b.add_argument("--encoder", default=DEFAULT_ENCODER)
    b.add_argument("--raw-terms", action="store_true", help="sirovi pojmovi umjesto ESCO labela")

    q = sub.add_parser("query", help="vraća RetrievalResult za kandidate")
    q.add_argument("--candidates", required=True, help="NormalizedProfile kandidata")
    q.add_argument("--index", default="faiss_index")
    q.add_argument("--k", type=int, default=50, help="broj oglasa po kandidatu")
    q.add_argument("--out", help="izlazna .json datoteka (bez toga ispis na ekran)")

    args = parser.parse_args()

    if args.command == "build":
        jobs = load_profiles(args.jobs, entity_type="job")
        job_index = JobIndex.build(jobs, args.encoder, use_esco=not args.raw_terms)
        job_index.save(args.index)
        print(f"Indeksirano {len(jobs)} oglasa ({args.encoder}) -> {args.index}")

    elif args.command == "query":
        candidates = load_profiles(args.candidates, entity_type="candidate")
        job_index = JobIndex.load(args.index)
        results = [retrieve(c, job_index, args.k) for c in candidates]
        if args.out:
            save_json(results, args.out)
            print(f"Spremljeno {len(results)} RetrievalResult objekata u {args.out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
