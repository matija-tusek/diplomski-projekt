import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # isti model kao Grupa 3
# smece je islo do 0.76 (Docker -> "dock operations"), a dobri pogoci su uglavnom od 0.78
EMBEDDING_CUTOFF = 0.78

model = None  # ucitava se samo jednom, kad prvi put zatreba


def get_model():
    global model
    if model is None:
        # graficka ako je ima i ako je torch instaliran s CUDA podrskom, inace procesor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding model running on {device}")
        model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# sto se sve embeda za svaki ESCO koncept:
#   "labels"      -> sluzbeni naziv + sinonimi (zadano)
#   "preferred"   -> samo sluzbeni naziv
#   "description" -> sluzbeni naziv + sinonimi + opis koncepta
VARIANTS = ["labels", "preferred", "description"]


def cache_folder(csv_path, variant="labels"):
    # data/esco/skills_en.csv -> data/esco/embeddings/skills_en
    name = Path(csv_path).stem
    if variant != "labels":
        name += "_" + variant
    return Path(csv_path).parent / "embeddings" / name


def read_labels(csv_path, variant="labels"):
    labels = []
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uri = (row.get("conceptUri") or "").strip()
            preferred = (row.get("preferredLabel") or "").strip()
            if not uri or not preferred:
                continue

            texts = [preferred]
            if variant != "preferred":
                texts += (row.get("altLabels") or "").splitlines()
            if variant == "description":
                texts.append(row.get("description") or "")

            seen = set()
            for text in texts:
                text = text.strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    labels.append({"uri": uri, "preferred_label": preferred, "label": text})
    return labels


def build_embeddings(csv_path, variant="labels"):
    labels = read_labels(csv_path, variant)
    print(f"Embedding {len(labels)} labels from {Path(csv_path).name}, this can take a few minutes...")
    texts = [label["label"] for label in labels]
    vectors = get_model().encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)

    folder = cache_folder(csv_path, variant)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", vectors.astype(np.float16))  # float16 zauzima pola manje mjesta
    meta = {"model": MODEL_NAME, "csv_hash": file_hash(csv_path), "labels": labels}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {folder}")


def load_embeddings(csv_path, variant="labels"):
    folder = cache_folder(csv_path, variant)
    meta_path = folder / "meta.json"

    meta = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # nema cachea, ili je napravljen s drugim modelom / drugom verzijom CSV-a
    if meta is None or meta["model"] != MODEL_NAME or meta["csv_hash"] != file_hash(csv_path):
        build_embeddings(csv_path, variant)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    vectors = np.load(folder / "vectors.npy").astype(np.float32)
    print(f"ESCO embeddings: loaded {len(vectors)} vectors from {folder}")
    return {"vectors": vectors, "labels": meta["labels"]}


def embedding_lookup(term, embeddings, top_k=5):
    query = get_model().encode([term], normalize_embeddings=True)[0]
    scores = embeddings["vectors"] @ query  # vektori su normalizirani pa je ovo kosinusna slicnost

    results = []  # (concept, esco_label, score), od najslicnijeg prema dolje
    seen_uris = set()
    for i in np.argsort(-scores):
        label = embeddings["labels"][i]
        # isti koncept zna imati vise sinonima pri vrhu, brojimo ga samo jednom
        if label["uri"] in seen_uris:
            continue
        seen_uris.add(label["uri"])
        concept = {"uri": label["uri"], "preferred_label": label["preferred_label"]}
        results.append((concept, label["label"], float(scores[i])))
        if len(results) == top_k:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description="Embed all ESCO labels once and save them to disk")
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--occupations", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="labels", help="what to embed for each ESCO concept")
    args = parser.parse_args()

    build_embeddings(args.skills, args.variant)
    if args.occupations:
        build_embeddings(args.occupations, args.variant)


if __name__ == "__main__":
    main()
