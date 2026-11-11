"""Data models for Group 1 extraction.

Two layers:

1. LLM models (`Llm*`): what the model is asked to fill in. They are passed to Ollama as a JSON
   schema, so the model can only produce matching JSON. Evidence is a plain quote here, and it is
   the FIRST property of every item: the model writes the quote, then the value read from it.
   Offsets, IDs and the raw text are not the model's job; code adds them.

2. Contract models (`CandidateProfile`, `JobProfile`): the shared schema v1.0 from
   `schemas/project_schemas.json`. The final output is validated against them before it is saved.
   "Unknown is not false": anything the text does not state is null, never a default.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Modality = Literal["required", "preferred", "unspecified"]
RemoteMode = Literal["onsite", "hybrid", "remote", "any"]
JobType = Literal["internship", "junior", "other"]
EducationLevel = Literal["high_school", "bsc", "msc", "phd"]
EducationStatus = Literal["enrolled", "graduated", "dropped_out"]
LanguageLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native"]


# --------------------------------------------------------------------------- #
# 1. LLM models (what Qwen fills in)
# --------------------------------------------------------------------------- #

class LlmStudentStatus(BaseModel):
    evidence: str | None
    is_student: bool | None


class LlmEducation(BaseModel):
    evidence: str
    level: EducationLevel | None
    field: str | None
    institution: str | None
    status: EducationStatus | None
    expected_graduation: str | None


class LlmSkills(BaseModel):
    """One quote can name several skills ("pandas and scikit-learn"); code expands them."""
    evidence: str
    names: list[str]


class LlmExperience(BaseModel):
    evidence: str
    title: str
    organization: str | None
    start: str | None
    end: str | None
    stated_duration_months: int | None
    description: str | None
    skills_used: list[str]


class LlmLanguage(BaseModel):
    evidence: str
    language: str
    level: LanguageLevel | None


class LlmPreferences(BaseModel):
    target_roles: list[str]
    interests: list[str]
    locations: list[str]
    remote_mode: RemoteMode | None
    job_types: list[JobType]


class LlmCandidate(BaseModel):
    student_status: LlmStudentStatus
    education: list[LlmEducation]
    experience: list[LlmExperience]
    skills: list[LlmSkills]
    languages: list[LlmLanguage]
    preferences: LlmPreferences


class LlmRemoteMode(BaseModel):
    evidence: str | None
    value: RemoteMode | None
    modality: Modality


class LlmJobSkills(BaseModel):
    evidence: str
    names: list[str]
    modality: Modality


class LlmJobEducation(BaseModel):
    evidence: str | None
    level: EducationLevel | None
    field: str | None
    modality: Modality


class LlmStudentRequired(BaseModel):
    evidence: str | None
    value: bool | None
    modality: Modality


class LlmJobExperience(BaseModel):
    evidence: str | None
    min_years: float | None
    modality: Modality


class LlmJobLanguage(BaseModel):
    evidence: str
    language: str
    level: LanguageLevel | None
    modality: Modality


class LlmLocation(BaseModel):
    city: str | None
    country: str | None


class LlmJob(BaseModel):
    title: str
    occupation: str | None
    company: str | None
    job_type: JobType
    location: LlmLocation
    remote_mode: LlmRemoteMode
    skills: list[LlmJobSkills]
    education: LlmJobEducation
    student_status_required: LlmStudentRequired
    experience: LlmJobExperience
    languages: list[LlmJobLanguage]


# --------------------------------------------------------------------------- #
# 2. Contract models (shared schema v1.0, final output)
# --------------------------------------------------------------------------- #

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(Strict):
    text: str  # always a verbatim slice of the source: text == raw_text[char_start:char_end]
    source_field: Literal["cv_text", "job_text"]
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class CandidateSource(Strict):
    type: Literal["cv"] = "cv"
    raw_text: str


class StudentStatus(Strict):
    is_student: bool | None
    evidence: Evidence | None


class CandidateEducation(Strict):
    level: EducationLevel | None
    field: str | None
    institution: str | None
    status: EducationStatus | None
    expected_graduation: str | None = Field(pattern=r"^\d{4}(-(0[1-9]|1[0-2]))?$")
    evidence: Evidence


class CandidateSkill(Strict):
    name: str
    evidence: Evidence


class CandidateExperience(Strict):
    title: str
    organization: str | None
    months: int | None = Field(ge=0)
    description: str | None
    evidence: Evidence


class CandidateLanguage(Strict):
    language: str
    level: LanguageLevel | None
    evidence: Evidence


class Preferences(Strict):
    target_roles: list[str] | None
    interests: list[str] | None
    locations: list[str] | None
    remote_mode: RemoteMode | None
    job_types: list[JobType] | None


class CandidateProfile(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    candidate_id: str
    source: CandidateSource
    student_status: StudentStatus
    education: list[CandidateEducation]
    skills: list[CandidateSkill]
    experience: list[CandidateExperience]
    languages: list[CandidateLanguage]
    preferences: Preferences


class JobSource(Strict):
    dataset: str
    raw_text: str


class Location(Strict):
    city: str | None
    country: str | None  # ISO 3166-1 alpha-2, e.g. "HR"


class JobRemoteMode(Strict):
    value: RemoteMode
    modality: Modality
    evidence: Evidence


class JobSkill(Strict):
    name: str
    modality: Modality
    evidence: Evidence


class JobEducation(Strict):
    level: EducationLevel | None
    field: str | None
    modality: Modality
    evidence: Evidence


class JobStudentRequired(Strict):
    value: bool
    modality: Modality
    evidence: Evidence


class JobExperience(Strict):
    min_years: float = Field(ge=0)
    modality: Modality
    evidence: Evidence


class JobLanguage(Strict):
    language: str
    level: LanguageLevel | None
    modality: Modality
    evidence: Evidence


class JobProfile(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    job_id: str
    source: JobSource
    title: str
    occupation: str | None
    company: str | None
    job_type: JobType
    location: Location
    remote_mode: JobRemoteMode | None
    skills: list[JobSkill]
    education: JobEducation | None
    student_status_required: JobStudentRequired | None
    experience: JobExperience | None
    languages: list[JobLanguage]
