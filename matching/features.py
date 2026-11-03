import schemas.models as models
from random import random

# TODO: prilagoditi funkciju za score
def calculate_score(features : models.MatchFeatures):
    return features.semantic_similarity

# TODO: pravilno izracunati vrijednosti za feature
def build_features(candidate: models.NormalizedProfile, job: models.NormalizedProfile, semantic_similarity : float):
    req_cov, missing_req = skill_coverage(candidate, job, modality="required")
    pref_cov, missing_pref = skill_coverage(candidate, job, modality="preferred")

    return models.MatchFeatures(
        semantic_similarity=semantic_similarity,
        required_skill_coverage=req_cov,
        preferred_skill_coverage=pref_cov,
        number_missing_required_skills=len(missing_req),
        number_missing_preferred_skills=len(missing_pref),
        missing_required_skills=missing_req,
        missing_preferred_skills=missing_pref,
        esco_occupation_similarity=occupation_similarity(candidate, job),
        education_match=education_match(candidate, job),
        seniority_match=seniority_match(candidate, job),
        experience_gap=experience_gap(candidate, job),
        language_match=language_match(candidate, job),
        location_match=location_match(candidate, job),
        remote_preference_match=remote_preference_match(candidate, job),
    )

def skill_coverage(cand: models.NormalizedProfile, job: models.NormalizedProfile, modality: str) -> tuple[float | None, list[str]]:
    """Share of the job's skills with this modality that the candidate has.
    Returns (coverage, missing). Job has no such skills -> (None, [])."""
    # TODO
    return (random(), [])


def occupation_similarity(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Closeness of the candidate's target role to the job's occupation."""
    # TODO
    return random()


def education_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Does the candidate's education level meet the required one (EDUCATION_ORDER)?"""
    # TODO
    return random()


def seniority_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Fit of the candidate to job_type (internship / junior / other)."""
    # TODO
    return random()


def experience_gap(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Required years minus candidate years. Note: 0 months != experience not stated."""
    # TODO
    return random()


def language_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Languages the job asks for vs. the candidate's levels (LANGUAGE_ORDER)."""
    # TODO
    return random()


def location_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Job city/country vs. candidate.profile.preferences.locations."""
    # TODO
    return random()


def remote_preference_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Candidate remote_mode preference vs. job remote_mode ('any' fits everything)."""
    # TODO
    return random()