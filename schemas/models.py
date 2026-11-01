"""Pydantic (v2) models generated from project_schemas.json (schema_version 1.0)."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Modality(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    UNSPECIFIED = "unspecified"


class RemoteMode(str, Enum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    ANY = "any"


class JobType(str, Enum):
    INTERNSHIP = "internship"
    JUNIOR = "junior"
    OTHER = "other"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    BSC = "bsc"
    MSC = "msc"
    PHD = "phd"


class EducationStatus(str, Enum):
    ENROLLED = "enrolled"
    GRADUATED = "graduated"
    DROPPED_OUT = "dropped_out"


class LanguageLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"
    NATIVE = "native"


class MappingStatus(str, Enum):
    EXACT = "EXACT"
    RELATED = "RELATED"
    UNMAPPED = "UNMAPPED"


class MappingMethod(str, Enum):
    ALIAS_LOOKUP = "alias_lookup"
    EXACT_LOOKUP = "exact_lookup"
    FUZZY_MATCH = "fuzzy_match"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    NONE = "none"


class ConstraintStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EntityType(str, Enum):
    CANDIDATE = "candidate"
    JOB = "job"


# --------------------------------------------------------------------------- #
# Shared objects
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    """Source-text evidence; attached to every extracted item (Group 1)."""

    text: str
    source_field: Literal["cv_text", "job_text"]
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> "Evidence":
        if self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        return self


# --------------------------------------------------------------------------- #
# CandidateProfile (Group 1 -> Group 2)
# --------------------------------------------------------------------------- #
class CandidateSource(BaseModel):
    type: str = "cv"
    raw_text: str


class StudentStatus(BaseModel):
    is_student: Optional[bool] = None  # None = unknown (not the same as False)
    evidence: Optional[Evidence] = None

    @model_validator(mode="after")
    def _evidence_if_known(self) -> "StudentStatus":
        if self.is_student is not None and self.evidence is None:
            raise ValueError("evidence is required when is_student is known")
        return self


class CandidateEducation(BaseModel):
    level: EducationLevel
    field: Optional[str] = None
    institution: Optional[str] = None
    status: EducationStatus
    expected_graduation: Optional[str] = Field(
        default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM"
    )
    evidence: Evidence


class CandidateSkill(BaseModel):
    name: str
    evidence: Evidence


class CandidateExperience(BaseModel):
    title: str
    organization: Optional[str] = None
    months: Optional[int] = Field(default=None, ge=0)
    description: Optional[str] = None
    evidence: Evidence


class CandidateLanguage(BaseModel):
    language: str
    level: LanguageLevel
    evidence: Evidence


class CandidatePreferences(BaseModel):
    target_roles: Optional[list[str]] = None
    interests: Optional[list[str]] = None
    locations: Optional[list[str]] = None
    remote_mode: Optional[RemoteMode] = None
    job_types: Optional[list[JobType]] = None


class CandidateProfile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    source: CandidateSource
    student_status: StudentStatus
    education: list[CandidateEducation] = Field(default_factory=list)
    skills: list[CandidateSkill] = Field(default_factory=list)
    experience: list[CandidateExperience] = Field(default_factory=list)
    languages: list[CandidateLanguage] = Field(default_factory=list)
    preferences: Optional[CandidatePreferences] = None


# --------------------------------------------------------------------------- #
# JobProfile (Group 1 -> Group 2)
# Rule: every requirement carries value + modality + evidence.
# --------------------------------------------------------------------------- #
class JobSource(BaseModel):
    dataset: str
    raw_text: str


class JobLocation(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None  # ISO 3166-1 alpha-2, e.g. "HR"


class JobRemoteMode(BaseModel):
    value: RemoteMode
    modality: Modality
    evidence: Evidence


class JobSkill(BaseModel):
    name: str
    modality: Modality
    evidence: Evidence


class JobEducation(BaseModel):
    level: EducationLevel
    field: Optional[str] = None
    modality: Modality
    evidence: Evidence


class JobStudentStatusRequired(BaseModel):
    value: bool
    modality: Modality
    evidence: Evidence


class JobExperience(BaseModel):
    min_years: float = Field(ge=0)
    modality: Modality
    evidence: Evidence


class JobLanguage(BaseModel):
    language: str
    level: LanguageLevel
    modality: Modality
    evidence: Evidence


class JobProfile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str
    source: JobSource
    title: str
    occupation: Optional[str] = None
    company: Optional[str] = None
    job_type: JobType
    location: Optional[JobLocation] = None
    remote_mode: Optional[JobRemoteMode] = None
    skills: list[JobSkill] = Field(default_factory=list)
    education: Optional[JobEducation] = None
    student_status_required: Optional[JobStudentStatusRequired] = None
    experience: Optional[JobExperience] = None
    languages: list[JobLanguage] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Normalization (Group 2 -> Groups 3 & 4)
# --------------------------------------------------------------------------- #
class EscoConcept(BaseModel):
    uri: str
    preferred_label: str


class Mapping(BaseModel):
    status: MappingStatus
    method: MappingMethod
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizedTerm(BaseModel):
    """Same shape for skills and occupations.

    The original term is always kept; esco_concept is None when the status is UNMAPPED.
    modality/evidence are optional (the occupations example has neither), with no invented default.
    """

    source_term: str
    normalized_label: str
    esco_concept: Optional[EscoConcept] = None
    mapping: Mapping
    modality: Optional[Modality] = None
    evidence: Optional[Evidence] = None

    @model_validator(mode="after")
    def _check_unmapped(self) -> "NormalizedTerm":
        if self.mapping.status == MappingStatus.UNMAPPED and self.esco_concept is not None:
            raise ValueError("esco_concept must be None when mapping.status is UNMAPPED")
        return self


NormalizedSkill = NormalizedTerm
NormalizedOccupation = NormalizedTerm


class Normalized(BaseModel):
    skills: list[NormalizedSkill] = Field(default_factory=list)
    occupations: list[NormalizedOccupation] = Field(default_factory=list)


class NormalizedProfile(BaseModel):
    """The original profile stays untouched under 'profile'; Group 2 only adds 'normalized'."""

    schema_version: str = SCHEMA_VERSION
    entity_type: EntityType
    entity_id: str
    profile: Union[CandidateProfile, JobProfile]
    normalized: Normalized

    @model_validator(mode="after")
    def _check_consistency(self) -> "NormalizedProfile":
        if self.entity_type == EntityType.CANDIDATE:
            ok = isinstance(self.profile, CandidateProfile) and self.profile.candidate_id == self.entity_id
        else:
            ok = isinstance(self.profile, JobProfile) and self.profile.job_id == self.entity_id
        if not ok:
            raise ValueError("entity_type/entity_id do not match the embedded profile")
        return self


# --------------------------------------------------------------------------- #
# Retrieval (Group 3)
# --------------------------------------------------------------------------- #
class RetrievalDocument(BaseModel):
    entity_type: EntityType
    entity_id: str
    retrieval_text: str
    encoder_model: str
    embedding_dim: int = Field(gt=0)


class RetrievalHit(BaseModel):
    job_id: str
    semantic_similarity: float
    retrieval_rank: int = Field(ge=1)


class RetrievalResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    encoder_model: str
    k: int = Field(gt=0)
    results: list[RetrievalHit]


# --------------------------------------------------------------------------- #
# Matching (Group 4)
# --------------------------------------------------------------------------- #
class MatchFeatures(BaseModel):
    semantic_similarity: float
    required_skill_coverage: float = Field(ge=0.0, le=1.0)
    preferred_skill_coverage: float = Field(ge=0.0, le=1.0)
    number_missing_required_skills: int = Field(ge=0)
    number_missing_preferred_skills: int = Field(ge=0)
    missing_required_skills: list[str] = Field(default_factory=list)
    missing_preferred_skills: list[str] = Field(default_factory=list)
    esco_occupation_similarity: float
    education_match: float
    seniority_match: float
    experience_gap: float
    language_match: float
    location_match: float
    remote_preference_match: float


class ConstraintCheck(BaseModel):
    constraint: str  # e.g. "required_skills", "student_status", "education", ...
    status: ConstraintStatus
    required: bool
    value: dict[str, Any] = Field(default_factory=dict)


class Constraints(BaseModel):
    overall_status: ConstraintStatus
    checks: list[ConstraintCheck]


class Explanation(BaseModel):
    language: str  # e.g. "hr"
    llm_model: str
    text: str


class MatchResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    job_id: str
    features: MatchFeatures
    match_score: float = Field(ge=0.0, le=1.0)
    score_weights_version: str
    constraints: Constraints
    eligible: bool
    explanation: Optional[Explanation] = None

    @model_validator(mode="after")
    def _check_eligible(self) -> "MatchResult":
        # eligible = false if any required check FAILs (UNKNOWN does not exclude)
        expected = not any(
            c.required and c.status == ConstraintStatus.FAIL for c in self.constraints.checks
        )
        if self.eligible != expected:
            raise ValueError(
                f"eligible={self.eligible} contradicts constraints (expected {expected})"
            )
        return self


# --------------------------------------------------------------------------- #
# Final pipeline output (Group 4)
# --------------------------------------------------------------------------- #
class Recommendation(BaseModel):
    final_rank: int = Field(ge=1)
    match: MatchResult  # eligible == True

    @model_validator(mode="after")
    def _must_be_eligible(self) -> "Recommendation":
        if not self.match.eligible:
            raise ValueError("recommendations may only contain eligible matches")
        return self


class ExcludedJob(BaseModel):
    job_id: str
    reason: str
    match: MatchResult  # eligible == False

    @model_validator(mode="after")
    def _must_be_ineligible(self) -> "ExcludedJob":
        if self.match.eligible:
            raise ValueError("excluded entries must have eligible = False")
        return self


class RecommendationResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    generated_at: datetime
    recommendations: list[Recommendation] = Field(default_factory=list)
    excluded: list[ExcludedJob] = Field(default_factory=list)