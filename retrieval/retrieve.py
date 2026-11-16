"""Glavna funkcija za Grupu 4: kandidat -> RetrievalResult."""

SCHEMA_VERSION = "1.0"


def retrieve(candidate, job_index, k=50):
    hits = job_index.search(candidate, k)
    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["entity_id"],
        "encoder_model": job_index.encoder_name,
        "k": len(hits),
        "results": [
            {"job_id": job_id, "semantic_similarity": round(score, 4), "retrieval_rank": rank}
            for rank, (job_id, score) in enumerate(hits, start=1)
        ],
    }
    validate_retrieval_result(result)
    return result


def validate_retrieval_result(result):
    """Osnovna provjera da izlaz odgovara shemi RetrievalResult."""
    for key in ("schema_version", "candidate_id", "encoder_model", "k", "results"):
        if key not in result:
            raise ValueError(f"RetrievalResult nema polje '{key}'")
    if result["k"] != len(result["results"]):
        raise ValueError("k ne odgovara broju rezultata")
    for expected_rank, r in enumerate(result["results"], start=1):
        if r["retrieval_rank"] != expected_rank:
            raise ValueError("retrieval_rank nije redom 1..k")
        if not -1.0 <= r["semantic_similarity"] <= 1.0001:
            raise ValueError(f"neispravna sličnost za {r['job_id']}")
