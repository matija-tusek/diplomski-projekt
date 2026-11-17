"""Run from the repo root: python -m matching.test_features"""
from pathlib import Path
from types import SimpleNamespace

from pydantic import TypeAdapter

import schemas.models as models
from matching.constraints import check_constraints, exclusion_reason
from matching.features import (
    build_features, calculate_score, education_match, experience_gap, language_match, location_match, occupation_similarity,
    remote_preference_match, seniority_match, skill_coverage,
)


# --- skill_coverage on hand-made skills -------------------------------------
def skill(name, modality="unspecified", status="EXACT", uri=None):
    esco = {"uri": uri, "preferred_label": "x"} if uri else None
    return models.NormalizedTerm(
        source_term=name, normalized_label=name, esco_concept=esco, modality=modality,
        mapping={"status": status, "method": "alias_lookup", "confidence": 1.0},
    )


def skills(*items):
    # skill_coverage only reads .normalized.skills
    return SimpleNamespace(normalized=models.Normalized(skills=list(items)))


cand = skills(skill("python"), skill("SQL"), skill("FastAPI", status="RELATED", uri="esco/web"))
job = skills(
    skill("Python", "required"), skill("SQL", "required"), skill("SQL", "required"),  # duplicate
    skill("Docker", "preferred"),
    skill("Django", "preferred", status="RELATED", uri="esco/web"),  # same broad URI as FastAPI
)

assert skill_coverage(cand, job, "required") == (1.0, [])            # case-insensitive, duplicate counted once
assert skill_coverage(cand, job, "preferred") == (0.0, ["Docker", "Django"])  # RELATED URI is not a match
assert skill_coverage(cand, skills(), "required") == (1.0, [])       # job asks nothing
exact = skills(skill("Postgres", "required", uri="esco/sql"))
assert skill_coverage(skills(skill("PostgreSQL", uri="esco/sql")), exact, "required") == (1.0, [])  # EXACT URI matches


# --- other features on the mock data (scenarios from generate_mock_data.py) --
data = Path(__file__).parent / "mock_data"
load = TypeAdapter(list[models.NormalizedProfile]).validate_json
C = {p.entity_id: p for p in load((data / "candidates.json").read_bytes())}
J = {p.entity_id: p for p in load((data / "jobs.json").read_bytes())}
c1, c2, c3 = C["cand_001"], C["cand_002"], C["cand_003"]

assert occupation_similarity(c1, J["job_002"]) == 0.93   # data scientist (1.0) x junior data scientist (0.93)
assert occupation_similarity(c1, J["job_003"]) == 0.0    # data scientist vs software developer
assert education_match(c2, J["job_001"]) == 0.0          # BSc vs MSc required
assert education_match(c1, J["job_003"]) == 1.0          # job states no education
assert seniority_match(c2, J["job_001"]) == 0.0          # wants junior, job is internship
assert experience_gap(c1, J["job_001"]) == 1.0           # 2 years asked, 12 months listed
assert experience_gap(c3, J["job_001"]) == 2.0           # no experience listed = 0 years
assert language_match(c3, J["job_004"]) == 0.0           # English B1 < B2
assert language_match(c1, J["job_005"]) == 0.5           # English ok, no German
assert language_match(c3, J["job_006"]) == 1.0           # preferred German is ignored
assert location_match(c1, J["job_004"]) == 0.0           # Split onsite, wants Zagreb
assert location_match(c1, J["job_005"]) == 1.0           # fully remote job
assert remote_preference_match(c1, J["job_002"]) == 0.5  # hybrid vs onsite
assert remote_preference_match(c3, J["job_001"]) is None # no preference stated


# --- unknown stays unknown -------------------------------------------------
def with_profile(p, **changes):
    return SimpleNamespace(profile=p.profile.model_copy(update=changes), normalized=p.normalized)


no_months = c1.profile.experience[0].model_copy(update={"months": None})
assert experience_gap(with_profile(c1, experience=[no_months]), J["job_001"]) is None
no_level = c3.profile.languages[0].model_copy(update={"level": None})  # English, level unknown
assert language_match(with_profile(c3, languages=[no_level]), J["job_001"]) is None

# --- calculate_score -------------------------------------------------------
perfect = models.MatchFeatures(
    semantic_similarity=1.0, required_skill_coverage=1.0, preferred_skill_coverage=1.0,
    number_missing_required_skills=0, number_missing_preferred_skills=0,
    esco_occupation_similarity=1.0, education_match=1.0, seniority_match=1.0, experience_gap=0.0,
    language_match=1.0, location_match=1.0, remote_preference_match=1.0,
)
assert calculate_score(perfect) == 1.0
assert calculate_score(perfect.model_copy(update={"remote_preference_match": None})) == 1.0  # unknown is not 0
assert calculate_score(perfect.model_copy(update={"remote_preference_match": 0.0})) == 0.95
assert calculate_score(perfect.model_copy(update={"experience_gap": 1.0})) == 0.95          # fit 0.5 x weight 0.10
assert calculate_score(perfect.model_copy(update={"semantic_similarity": -3.0})) == 0.0      # clamped

# --- constraints ------------------------------------------------------------
def constraints(c, job_id):
    return check_constraints(c, J[job_id], build_features(c, J[job_id], 0.5))


k = constraints(c1, "job_001")  # same statuses as the MatchResult example in project_schemas.json
assert [ch.status.value for ch in k.checks] == ["PASS", "PASS", "PASS", "NOT_APPLICABLE", "PASS", "PASS"]
assert k.overall_status == models.ConstraintStatus.PASS
assert exclusion_reason(constraints(c2, "job_005")) == "constraint FAIL: language (German B2 required)"
assert exclusion_reason(constraints(c1, "job_008")) == "constraint FAIL: required_skills (missing Excel)"
assert exclusion_reason(constraints(c2, "job_001")) == (
    "constraint FAIL: required_skills (missing Python); student_status (students only); education (msc required)")
assert constraints(c3, "job_006").overall_status == models.ConstraintStatus.UNKNOWN  # student status unknown: not excluded

print("features OK")
