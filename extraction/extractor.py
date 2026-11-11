"""Turn a raw CV / job ad into a CandidateProfile / JobProfile (shared schema v1.0).

Steps for one document:
1. Ollama + Qwen3-4B fills the Llm* schema (structured output, temperature 0).
2. Every evidence quote is located in the source text (evidence.locate). Items whose quote
   cannot be found are dropped and reported in `warnings`, because they may be hallucinated.
3. Values the LLM is bad at (experience months, empty lists vs. unknown) are derived in code.
4. The result is validated against the contract model before it is returned.
"""
import os
import re
from datetime import date

import ollama

from .evidence import locate
from .prompts import CV_PROMPT, JOB_PROMPT
from .schemas import (
    SCHEMA_VERSION,
    CandidateProfile,
    JobProfile,
    LlmCandidate,
    LlmJob,
    LlmSkills,
)

MODEL = os.environ.get("EXTRACTION_MODEL", "qwen3:4b-instruct")
OPTIONS = {"temperature": 0, "num_ctx": 8192}


def call_llm(system_prompt, text, schema):
    """One structured-output call. Returns an instance of `schema`."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        format=schema.model_json_schema(),
        options=OPTIONS,
    )
    return schema.model_validate_json(response.message.content)


class _Grounder:
    """Locates evidence for one document and records what had to be dropped."""

    def __init__(self, source, source_field):
        self.source = source
        self.source_field = source_field
        self.found = 0
        self.dropped = []

    def __call__(self, quote, what):
        evidence = locate(quote, self.source, self.source_field)
        if evidence is None:
            self.dropped.append({"item": what, "quote": quote})
        else:
            self.found += 1
        return evidence

    def report(self):
        return {"evidence_found": self.found, "dropped": self.dropped}


def _months(start, end, stated, today=None):
    """Experience length in months, or None when the text is not precise enough.

    "2025-03" .. "2025-08" counts both ends (6 months). Year-only dates give None
    ("2024 - 2025" could be 2 or 24 months: unknown is not false).
    """
    if stated is not None:
        return stated
    today = today or date.today()
    ym = re.compile(r"^(\d{4})-(\d{2})$")

    def parse(value):
        if value and value.strip().lower() in ("present", "now", "danas", "sada"):
            return today.year, today.month
        m = ym.match(value or "")
        return (int(m.group(1)), int(m.group(2))) if m else None

    s, e = parse(start), parse(end)
    if not s or not e:
        return None
    months = (e[0] - s[0]) * 12 + (e[1] - s[1]) + 1
    return months if months > 0 else None


_NATIVE = re.compile(r"native|mother tongue|first language|materinsk|izvorn", re.I)
_NUMBER_OR_NEGATION = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|jedn\w*|dvij\w*|dva|tri|četiri|pet|šest"
    r"|no|not|without|none|bez|nije|nisu)\b", re.I)


def _grounded_level(level, evidence):
    """Keep a language level only if the evidence actually writes it ("fluent" is not C1).

    A sanity check, not proof: on a shared line "English (C1), German (A2)" a swapped level
    would still pass.
    """
    if level is None or evidence is None:
        return None
    if level == "native":
        return level if _NATIVE.search(evidence["text"]) else None
    pattern = rf"(?<![A-Za-z]){level[0]}\s*-?\s*{level[1]}(?!\d)"
    return level if re.search(pattern, evidence["text"], re.I) else None


_DEGREE = {
    "high_school": r"high school|secondary|gymnasium|gimnazij|srednj",
    "bsc": r"bachelor|\bb\.?sc|\bb\.?a\b|undergraduate|preddiplom|prvostup|bacc",
    "msc": r"master|\bm\.?sc|\bm\.?a\b|\bgraduate|(?<!pred)diplomsk|magist",
    "phd": r"ph\.?d|doctor|doktor",
}


def _grounded_degree(level, evidence):
    """Keep an education level only if the evidence names a degree ("student" is not a degree)."""
    if level is None or evidence is None:
        return None
    return level if re.search(_DEGREE[level], evidence["text"], re.I) else None


def _skills(ground, groups, with_modality):
    """Expand {evidence, names[, modality]} groups into one contract item per skill."""
    items, seen = [], set()
    for group in groups:
        names = [n.strip() for n in group.names if n.strip() and n.strip().lower() not in seen]
        if not names:
            continue
        ev = ground(group.evidence, "skills: " + ", ".join(names))
        if ev is None:
            continue
        for name in names:
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            item = {"name": name}
            if with_modality:
                item["modality"] = group.modality
            item["evidence"] = ev
            items.append(item)
    return items


def _or_none(values):
    """Empty list from the LLM means "not stated", which the contract writes as null."""
    return values or None


def _graduation(value):
    if value and re.fullmatch(r"\d{4}(-(0[1-9]|1[0-2]))?", value.strip()):
        return value.strip()
    return None


def extract_candidate(text, candidate_id, today=None):
    """CV text -> (CandidateProfile dict, report dict)."""
    llm = call_llm(CV_PROMPT, text, LlmCandidate)
    ground = _Grounder(text, "cv_text")

    status_evidence = None
    if llm.student_status.is_student is not None:
        status_evidence = ground(llm.student_status.evidence, "student_status")
    student_status = {
        "is_student": llm.student_status.is_student if status_evidence else None,
        "evidence": status_evidence,
    }

    education = []
    for e in llm.education:
        ev = ground(e.evidence, f"education: {e.level} {e.field}")
        if ev:
            education.append({
                "level": _grounded_degree(e.level, ev), "field": e.field, "institution": e.institution,
                "status": e.status, "expected_graduation": _graduation(e.expected_graduation),
                "evidence": ev,
            })

    experience = []
    for x in llm.experience:
        ev = ground(x.evidence, f"experience: {x.title}")
        if ev:
            experience.append({
                "title": x.title, "organization": x.organization,
                "months": _months(x.start, x.end, x.stated_duration_months, today),
                "description": x.description, "evidence": ev,
            })

    # Skills from the skills section, plus skills named in experience descriptions
    # (grounded on the description sentence, which the model copies word-for-word).
    from_experience = [LlmSkills(evidence=x.description, names=x.skills_used)
                       for x in llm.experience if x.description and x.skills_used]
    skills = _skills(ground, llm.skills + from_experience, with_modality=False)

    languages = []
    for lang in llm.languages:
        ev = ground(lang.evidence, f"language: {lang.language}")
        if ev:
            languages.append({"language": lang.language, "level": _grounded_level(lang.level, ev),
                              "evidence": ev})

    p = llm.preferences
    profile = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "source": {"type": "cv", "raw_text": text},
        "student_status": student_status,
        "education": education,
        "skills": skills,
        "experience": experience,
        "languages": languages,
        "preferences": {
            "target_roles": _or_none(p.target_roles),
            "interests": _or_none(p.interests),
            "locations": _or_none(p.locations),
            "remote_mode": p.remote_mode,
            "job_types": _or_none(list(dict.fromkeys(p.job_types))),
        },
    }
    return CandidateProfile.model_validate(profile).model_dump(), ground.report()


def _requirement(ground, item, what, **values):
    """A single-object job requirement, or None if the ad does not state it."""
    if all(v is None for v in values.values()):
        return None
    ev = ground(item.evidence, what)
    if ev is None:
        return None
    return {**values, "modality": item.modality, "evidence": ev}


def extract_job(text, job_id, dataset="manual"):
    """Job ad text -> (JobProfile dict, report dict)."""
    llm = call_llm(JOB_PROMPT, text, LlmJob)
    ground = _Grounder(text, "job_text")

    skills = _skills(ground, llm.skills, with_modality=True)

    languages = []
    for lang in llm.languages:
        ev = ground(lang.evidence, f"language: {lang.language}")
        if ev:
            languages.append({
                "language": lang.language, "level": _grounded_level(lang.level, ev),
                "modality": lang.modality, "evidence": ev,
            })

    experience = _requirement(ground, llm.experience, "experience", min_years=llm.experience.min_years)
    if experience and not _NUMBER_OR_NEGATION.search(experience["evidence"]["text"]):
        ground.found -= 1  # the quote exists, but it does not state a number of years
        ground.dropped.append({"item": "experience (no number in evidence)",
                               "quote": experience["evidence"]["text"]})
        experience = None

    education = _requirement(ground, llm.education, "education",
                             level=llm.education.level, field=llm.education.field)
    if education:
        education["level"] = _grounded_degree(education["level"], education["evidence"])
        if education["level"] is None and education["field"] is None:
            education = None

    country = llm.location.country.strip().upper() if llm.location.country else None
    profile = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "source": {"dataset": dataset, "raw_text": text},
        "title": llm.title,
        "occupation": llm.occupation,
        "company": llm.company,
        "job_type": llm.job_type,
        "location": {
            "city": llm.location.city,
            "country": country if country and re.fullmatch(r"[A-Z]{2}", country) else None,
        },
        "remote_mode": _requirement(ground, llm.remote_mode, "remote_mode",
                                    value=llm.remote_mode.value),
        "skills": skills,
        "education": education,
        "student_status_required": _requirement(ground, llm.student_status_required,
                                                "student_status_required",
                                                value=llm.student_status_required.value),
        "experience": experience,
        "languages": languages,
    }
    return JobProfile.model_validate(profile).model_dump(), ground.report()
