import schemas.models as models
import ollama

instructions_message = """
You explain to a student why a job was recommended to them. You get a list of facts about the match.
Write 2-3 sentences in plain prose and address the student as "you".
First name the skills they have that the job asks for, then the most important missing ones.
Then mention which job requirements they meet (and any that are unknown).
A missing preferred skill is a nice-to-have they lack, not a dealbreaker.
Use only the given facts. Do not mention scores or numbers.
"""


def match_facts(match: models.MatchResult, job: models.NormalizedProfile) -> str:
    """Plain-text facts for the LLM: which skills the candidate has or misses, which requirements pass."""
    f = match.features

    def skills(modality, missing):
        has = [s.source_term for s in job.normalized.skills if s.modality == modality and s.source_term not in missing]
        return ", ".join(dict.fromkeys(has)) or "none", ", ".join(missing) or "none"

    req_has, req_missing = skills(models.Modality.REQUIRED, f.missing_required_skills)
    pref_has, pref_missing = skills(models.Modality.PREFERRED, f.missing_preferred_skills)
    lines = [
        f"Job: {job.profile.title}",
        f"Required skills the candidate has: {req_has}",
        f"Required skills missing: {req_missing}",
        f"Preferred skills the candidate has: {pref_has}",
        f"Preferred skills missing: {pref_missing}",
    ]
    for status, label in [(models.ConstraintStatus.PASS, "met"),
                          (models.ConstraintStatus.FAIL, "not met"),
                          (models.ConstraintStatus.UNKNOWN, "unknown, the CV does not say")]:
        names = [c.constraint.replace("_", " ") for c in match.constraints.checks if c.status == status]
        if names:
            lines.append(f"Job requirements {label}: {', '.join(names)}")
    return "\n".join(lines)


def describe_match(match_result: models.MatchResult, job: models.NormalizedProfile, llm_model: str, language: str):
    response = ollama.chat(model=llm_model, messages=[
        {
            "role": "system",
            "content": instructions_message + f" Write only in the language with this ISO 639-1 code: {language}."
        },
        {
            "role": "user",
            "content": match_facts(match_result, job)
        }
    ])
    return response.message.content
