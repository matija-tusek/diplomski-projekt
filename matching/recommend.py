import schemas.models as models
from pydantic import TypeAdapter
from pathlib import Path
from datetime import datetime, timezone

from matching.features import build_features, calculate_score
from matching.explain import describe_match

LLM_MODEL = "qwen3:4b-instruct"
DATA_PATH = Path("matching/mock_data")
WEIGHTS_VERSION = "0.1"

jobs = TypeAdapter(list[models.NormalizedProfile]).validate_json((DATA_PATH / "jobs.json").read_bytes())
candidate = TypeAdapter(list[models.NormalizedProfile]).validate_json((DATA_PATH / "candidates.json").read_bytes())[0]
retrieval_result = TypeAdapter(list[models.RetrievalResult]).validate_json((DATA_PATH / "retrieval_results.json").read_bytes())[0]
matches = []

for r in retrieval_result.results:
    print(f"Calculating MatchResult object for: {r.job_id}")
    job = next((j for j in jobs if j.entity_id==r.job_id), None)
    features = build_features(candidate, job, r.semantic_similarity)
    score = calculate_score(features)
    match_res = models.MatchResult(
        candidate_id=candidate.entity_id,
        job_id=r.job_id,
        features=features,
        match_score=score,
        score_weights_version=WEIGHTS_VERSION,
        constraints={"overall_status": "PASS", "checks": []}, # TODO: dodati filtriranje po hard contraintovima
        eligible=True, # TODO: dodati filtriranje po hard contraintovima
        explanation=None,
    )
    match_res.explanation = models.Explanation(
        language="hr",
        llm_model=LLM_MODEL,
        text=describe_match(match_res, LLM_MODEL, "hr")
    )
    matches.append(match_res)

sorted_list = sorted(matches, key=lambda x: x.match_score)
filtered_list = [{"final_rank": i+1, "match":el} for i, el in enumerate(sorted_list[:5])]
recomendation = models.RecommendationResponse(
    candidate_id=candidate.entity_id,
    generated_at=datetime.now(timezone.utc).isoformat(),
    recommendations=filtered_list,
    excluded=[]
)
for el in recomendation.recommendations:
    print(f"rank: {el.final_rank}, {el.match.job_id}")
    print(el.match.explanation.text)
    print("------------")