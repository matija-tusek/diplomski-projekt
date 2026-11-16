"""FAISS indeks oglasa: gradi se jednom, sprema na disk i učitava za upite."""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from .text_builder import candidate_text, job_text

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "jobs.index"
META_FILE = "jobs_meta.json"


class JobIndex:
    """Embeddingi oglasa u FAISS IndexFlatIP.

    Vektori su normalizirani, pa je skalarni produkt jednak kosinusnoj sličnosti.
    """

    def __init__(self, model, encoder_name, index, documents, use_esco=True):
        self.model = model
        self.encoder_name = encoder_name
        self.index = index
        self.documents = documents      # RetrievalDocument po oglasu, istim redom kao u indeksu
        self.use_esco = use_esco

    @classmethod
    def build(cls, jobs, encoder_name=DEFAULT_ENCODER, use_esco=True, model=None):
        model = model or SentenceTransformer(encoder_name)
        texts = [job_text(j, use_esco) for j in jobs]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 100)

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        documents = [
            {
                "entity_type": "job",
                "entity_id": j["entity_id"],
                "retrieval_text": text,
                "encoder_model": encoder_name,
                "embedding_dim": int(emb.shape[1]),
            }
            for j, text in zip(jobs, texts)
        ]
        return cls(model, encoder_name, index, documents, use_esco)

    def search(self, candidate, k=50):
        """Vraća listu (job_id, sličnost), sortiranu od najsličnijeg."""
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []
        query = self.model.encode([candidate_text(candidate, self.use_esco)], normalize_embeddings=True)
        scores, positions = self.index.search(query, k)
        return [
            (self.documents[pos]["entity_id"], float(score))
            for score, pos in zip(scores[0], positions[0])
            if pos != -1
        ]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        meta = {"encoder_model": self.encoder_name, "use_esco": self.use_esco, "documents": self.documents}
        (directory / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory, model=None):
        directory = Path(directory)
        index = faiss.read_index(str(directory / INDEX_FILE))
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        model = model or SentenceTransformer(meta["encoder_model"])
        return cls(model, meta["encoder_model"], index, meta["documents"], meta["use_esco"])
