import schemas.models as models
from pydantic import TypeAdapter
from pathlib import Path
from datetime import datetime, timezone

from matching.features import SCORE_WEIGHTS_VERSION, build_features, calculate_score
from matching.constraints import check_constraints, exclusion_reason
from matching.explain import describe_match

LLM_MODEL = "qwen3:4b-instruct"
DATA_PATH = Path("matching/mock_data")

jobs = TypeAdapter(list[models.NormalizedProfile]).validate_json((DATA_PATH / "jobs.json").read_bytes())
candidate = TypeAdapter(list[models.NormalizedProfile]).validate_json((DATA_PATH / "candidates.json").read_bytes())[0]
retrieval_result = TypeAdapter(list[models.RetrievalResult]).validate_json((DATA_PATH / "retrieval_results.json").read_bytes())[0]
matches = []

for r in retrieval_result.results:
    print(f"Calculating MatchResult object for: {r.job_id}")
    job = next((j for j in jobs if j.entity_id==r.job_id), None)
    features = build_features(candidate, job, r.semantic_similarity)
    constraints = check_constraints(candidate, job, features)
    matches.append(models.MatchResult(
        candidate_id=candidate.entity_id,
        job_id=r.job_id,
        features=features,
        match_score=calculate_score(features),
        score_weights_version=SCORE_WEIGHTS_VERSION,
        constraints=constraints,
        eligible=constraints.overall_status != models.ConstraintStatus.FAIL,
        explanation=None,
    ))

matches.sort(key=lambda x: x.match_score, reverse=True)
top = [m for m in matches if m.eligible][:5]
for m in top:  # LLM only for what is shown: each call takes seconds
    job = next(j for j in jobs if j.entity_id == m.job_id)
    m.explanation = models.Explanation(language="en", llm_model=LLM_MODEL, text=describe_match(m, job, LLM_MODEL, "en"))

recomendation = models.RecommendationResponse(
    candidate_id=candidate.entity_id,
    generated_at=datetime.now(timezone.utc).isoformat(),
    recommendations=[{"final_rank": i, "match": m} for i, m in enumerate(top, start=1)],
    excluded=[{"job_id": m.job_id, "reason": exclusion_reason(m.constraints), "match": m}
              for m in matches if not m.eligible],
)
for el in recomendation.recommendations:
    print(f"rank: {el.final_rank}, {el.match.job_id}")
    print(el.match.explanation.text)
    print("------------")
for el in recomendation.excluded:
    print(f"excluded: {el.job_id}, {el.reason}")