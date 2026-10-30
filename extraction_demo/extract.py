"""Group 1 demo: turn a raw CV or job ad into structured JSON with a local LLM.

Ollama runs Qwen3-4B-Instruct locally. We pass a JSON schema (built from the
Pydantic models below) as `format`, so the model can only answer with JSON
that matches it. Afterwards we locate every evidence quote in the source text
and add char_start / char_end ourselves.

Usage:
    python extract.py cv  samples/cv.txt
    python extract.py job samples/job.txt
"""
import json
import sys
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel

MODEL = "qwen3:4b-instruct"

Modality = Literal["required", "preferred", "unspecified"]
LanguageLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native"]


class Evidence(BaseModel):
    text: str  # exact quote from the source; offsets are added in add_offsets()


# --- CandidateProfile (subset of project_schemas.json v1.0) ---

class Skill(BaseModel):
    name: str
    evidence: Evidence


class Language(BaseModel):
    language: str
    level: LanguageLevel | None
    evidence: Evidence


class Education(BaseModel):
    level: Literal["high_school", "bsc", "msc", "phd"] | None
    field: str | None
    institution: str | None
    status: Literal["enrolled", "graduated", "dropped_out"] | None
    evidence: Evidence


class CandidateProfile(BaseModel):
    is_student: bool | None
    education: list[Education]
    skills: list[Skill]
    languages: list[Language]


# --- JobProfile (subset): every requirement has value + modality + evidence ---

class JobSkill(Skill):
    modality: Modality


class JobLanguage(Language):
    modality: Modality


class JobProfile(BaseModel):
    title: str
    company: str | None
    job_type: Literal["internship", "junior", "other"]
    remote_mode: Literal["onsite", "hybrid", "remote", "any"] | None
    skills: list[JobSkill]
    languages: list[JobLanguage]


COMMON_RULES = """Rules:
- Extract only what the text states. If something is not stated, use null. Never guess.
- evidence.text must be copied word-for-word from the document (one sentence or phrase).
- Skill names are short raw terms as written in the text (e.g. "Python", "Docker")."""

CV_PROMPT = "You extract a structured candidate profile from a CV.\n" + COMMON_RULES

JOB_PROMPT = "You extract a structured job profile from a job advertisement.\n" + COMMON_RULES + """
- modality is "required" for must-haves (required, must, strong, needed),
  "preferred" for nice-to-haves (a plus, an advantage, preferred, nice to have),
  and "unspecified" when the ad does not say."""

KINDS = {
    "cv": (CandidateProfile, CV_PROMPT, "cv_text"),
    "job": (JobProfile, JOB_PROMPT, "job_text"),
}


def add_offsets(node, source, source_field):
    """Fill in source_field, char_start and char_end for every evidence quote.

    Done in code, not by the LLM, because LLMs can't count characters reliably.
    A quote not found verbatim gets null offsets: the model paraphrased or made it up.
    """
    if isinstance(node, list):
        for item in node:
            add_offsets(item, source, source_field)
    elif isinstance(node, dict):
        if "evidence" in node:
            ev = node["evidence"]
            start = source.find(ev["text"])
            ev["source_field"] = source_field
            ev["char_start"] = start if start >= 0 else None
            ev["char_end"] = start + len(ev["text"]) if start >= 0 else None
        for value in node.values():
            add_offsets(value, source, source_field)


def collect_evidence(node):
    if isinstance(node, list):
        return [ev for item in node for ev in collect_evidence(item)]
    if isinstance(node, dict):
        own = [node["evidence"]] if "evidence" in node else []
        return own + [ev for value in node.values() for ev in collect_evidence(value)]
    return []


def extract(kind, text):
    schema, prompt, source_field = KINDS[kind]
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        format=schema.model_json_schema(),
        options={"temperature": 0},
    )
    profile = schema.model_validate_json(response.message.content)
    data = profile.model_dump()
    add_offsets(data, text, source_field)
    return data


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in KINDS:
        sys.exit(__doc__)
    kind, path = sys.argv[1], Path(sys.argv[2])
    text = path.read_text(encoding="utf-8")

    data = extract(kind, text)

    out_path = Path("output") / f"{path.stem}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    evidence = collect_evidence(data)
    found = sum(ev["char_start"] is not None for ev in evidence)
    print(f"\nEvidence found verbatim in source: {found}/{len(evidence)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
