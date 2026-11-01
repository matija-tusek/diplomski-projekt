"""Generate mock inputs for Group 4 (ranking + explanation).

Group 4 consumes:
  - NormalizedProfile for candidates and jobs (from Group 2)
  - RetrievalResult per candidate (from Group 3)

Everything is built with the models from schemas.py, so evidence offsets are
computed from the raw text and every object is validated on creation.

Run:  python generate_mock_data.py
Out:  mock_data/candidates.json, mock_data/jobs.json, mock_data/retrieval_results.json
"""

import json
import re
from pathlib import Path

from schemas import (
    CandidateEducation,
    CandidateExperience,
    CandidateLanguage,
    CandidatePreferences,
    CandidateProfile,
    CandidateSkill,
    CandidateSource,
    EscoConcept,
    Evidence,
    JobEducation,
    JobExperience,
    JobLanguage,
    JobLocation,
    JobProfile,
    JobRemoteMode,
    JobSkill,
    JobSource,
    JobStudentStatusRequired,
    Mapping,
    Normalized,
    NormalizedProfile,
    NormalizedTerm,
    RetrievalHit,
    RetrievalResult,
    StudentStatus,
)

OUT_DIR = Path(__file__).parent / "mock_data"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ev(raw: str, phrase: str, field: str) -> Evidence:
    """Evidence whose char offsets are computed from the raw text."""
    start = raw.index(phrase)
    return Evidence(text=phrase, source_field=field, char_start=start, char_end=start + len(phrase))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# term (lowercase) -> (preferred_label, status, method, confidence)
SKILL_MAP = {
    "python": ("Python (computer programming)", "EXACT", "alias_lookup", 1.0),
    "sql": ("SQL", "EXACT", "exact_lookup", 1.0),
    "fastapi": ("web programming", "RELATED", "embedding_similarity", 0.81),
    "pytorch": ("machine learning", "RELATED", "embedding_similarity", 0.78),
    "docker": ("Docker", "EXACT", "alias_lookup", 0.98),
    "javascript": ("JavaScript", "EXACT", "exact_lookup", 1.0),
    "react": ("JavaScript framework", "RELATED", "embedding_similarity", 0.74),
    "node.js": ("JavaScript", "RELATED", "fuzzy_match", 0.88),
    "typescript": ("TypeScript", "EXACT", "exact_lookup", 1.0),
    "java": ("Java (computer programming)", "EXACT", "exact_lookup", 1.0),
    "spring": ("Java framework", "RELATED", "embedding_similarity", 0.72),
    "linux": ("Linux", "EXACT", "exact_lookup", 1.0),
    "kubernetes": ("Kubernetes", "EXACT", "exact_lookup", 1.0),
    "excel": ("use spreadsheets software", "EXACT", "alias_lookup", 0.97),
}

OCCUPATION_MAP = {
    "machine learning engineer": ("data scientist", "RELATED", "embedding_similarity", 0.77),
    "machine learning intern": ("data scientist", "RELATED", "embedding_similarity", 0.79),
    "data scientist": ("data scientist", "EXACT", "exact_lookup", 1.0),
    "junior data scientist": ("data scientist", "RELATED", "fuzzy_match", 0.93),
    "software developer": ("software developer", "EXACT", "exact_lookup", 1.0),
    "web developer": ("web developer", "EXACT", "exact_lookup", 1.0),
    "full-stack developer": ("software developer", "RELATED", "embedding_similarity", 0.83),
    "frontend developer": ("web developer", "RELATED", "embedding_similarity", 0.85),
    "java developer": ("software developer", "RELATED", "embedding_similarity", 0.84),
    "devops engineer": ("ICT system administrator", "RELATED", "embedding_similarity", 0.69),
    "data analyst": ("data analyst", "EXACT", "exact_lookup", 1.0),
}


def normalize(term, table, kind, modality=None, evidence=None) -> NormalizedTerm:
    hit = table.get(term.lower())
    if hit is None:  # UNMAPPED: keep the original term, no ESCO concept
        return NormalizedTerm(
            source_term=term,
            normalized_label=term,
            esco_concept=None,
            mapping=Mapping(status="UNMAPPED", method="none", confidence=0.0),
            modality=modality,
            evidence=evidence,
        )
    label, status, method, conf = hit
    return NormalizedTerm(
        source_term=term,
        normalized_label=term if kind == "skill" else label,
        esco_concept=EscoConcept(
            uri=f"http://data.europa.eu/esco/{kind}/mock-{slug(label)}", preferred_label=label
        ),
        mapping=Mapping(status=status, method=method, confidence=conf),
        modality=modality,
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
CANDIDATES = [
    # cand_001: MSc ML student (strong fit for ML/data jobs, no German, no Docker)
    dict(
        id="cand_001",
        student=(True, "MSc student, FER Zagreb (expected 2027)"),
        education=[("msc", "Computer Science", "FER Zagreb", "enrolled", "2027-07",
                    "MSc student, FER Zagreb (expected 2027)")],
        skills=[("Python", "Developed machine-learning models in Python"),
                ("FastAPI", "Built REST APIs using FastAPI"),
                ("SQL", "Wrote SQL queries for biomedical datasets"),
                ("PyTorch", "Trained neural networks in PyTorch")],
        experience=[("ML research assistant", "University lab", 12,
                     "Developed machine-learning models for biomedical time series",
                     "Research assistant (2025-2026): ML models for biomedical time series")],
        languages=[("English", "C1", "English (C1)"), ("Croatian", "native", "Croatian (native)")],
        prefs=dict(target_roles=["machine learning engineer", "data scientist"],
                   interests=["machine learning", "data science"], locations=["Zagreb"],
                   remote_mode="hybrid", job_types=["internship", "junior"]),
    ),
    # cand_002: graduated web developer (remote preference, 24 months experience)
    dict(
        id="cand_002",
        student=(False, "Graduated 2024, no longer a student"),
        education=[("bsc", "Computer Science", "University of Split", "graduated", None,
                    "BSc Computer Science, University of Split (2024)")],
        skills=[("JavaScript", "Built single-page apps in JavaScript"),
                ("React", "React front-end development"),
                ("Node.js", "REST backend with Node.js"),
                ("SQL", "PostgreSQL and SQL queries"),
                ("Docker", "Dockerised deployments")],
        experience=[("Junior web developer", "Agency d.o.o.", 24,
                     "Delivered client web applications",
                     "Junior web developer at Agency d.o.o. (2022-2024)")],
        languages=[("English", "B2", "English (B2)"), ("Croatian", "native", "Croatian (native)")],
        prefs=dict(target_roles=["full-stack developer", "frontend developer"],
                   interests=["web development"], locations=["Zagreb", "Split"],
                   remote_mode="remote", job_types=["junior"]),
    ),
    # cand_003: BSc student, unknown student status, sparse preferences, one UNMAPPED skill
    dict(
        id="cand_003",
        student=None,  # is_student unknown -> None, never an invented False
        education=[("bsc", "Computer Science", "University of Rijeka", "enrolled", "2027-07",
                    "BSc Computer Science, University of Rijeka (2027)")],
        skills=[("Java", "Java projects for university courses"),
                ("Spring", "Spring Boot web service"),
                ("Streamlit", "Prototype dashboards with Streamlit"),
                ("SQL", "Basic SQL")],
        experience=[],
        languages=[("English", "B1", "English (B1)"), ("German", "A2", "German (A2)"),
                   ("Croatian", "native", "Croatian (native)")],
        prefs=dict(target_roles=["java developer"], interests=None, locations=["Zagreb"],
                   remote_mode=None, job_types=["internship"]),
    ),
]


def build_candidate(c) -> NormalizedProfile:
    lines = ["Curriculum Vitae"]
    if c["student"] is not None:
        lines.append(c["student"][1])
    lines += [e[-1] for e in c["education"]]
    lines += [s[1] for s in c["skills"]]
    lines += [x[-1] for x in c["experience"]]
    lines += [lang[2] for lang in c["languages"]]
    raw = "\n".join(dict.fromkeys(lines))  # dedupe (student phrase == education phrase)
    f = "cv_text"

    if c["student"] is None:
        student = StudentStatus(is_student=None, evidence=None)
    else:
        student = StudentStatus(is_student=c["student"][0], evidence=ev(raw, c["student"][1], f))

    profile = CandidateProfile(
        candidate_id=c["id"],
        source=CandidateSource(type="cv", raw_text=raw),
        student_status=student,
        education=[CandidateEducation(level=l, field=fl, institution=i, status=st,
                                      expected_graduation=g, evidence=ev(raw, p, f))
                   for l, fl, i, st, g, p in c["education"]],
        skills=[CandidateSkill(name=n, evidence=ev(raw, p, f)) for n, p in c["skills"]],
        experience=[CandidateExperience(title=t, organization=o, months=m, description=d,
                                        evidence=ev(raw, p, f))
                    for t, o, m, d, p in c["experience"]],
        languages=[CandidateLanguage(language=la, level=lv, evidence=ev(raw, p, f))
                   for la, lv, p in c["languages"]],
        preferences=CandidatePreferences(**c["prefs"]),
    )
    normalized = Normalized(
        skills=[normalize(s.name, SKILL_MAP, "skill", "unspecified", s.evidence)
                for s in profile.skills],
        occupations=[normalize(r, OCCUPATION_MAP, "occupation")
                     for r in (profile.preferences.target_roles or [])],
    )
    return NormalizedProfile(entity_type="candidate", entity_id=c["id"],
                             profile=profile, normalized=normalized)


# --------------------------------------------------------------------------- #
# Jobs
# (phrase fields are exact sentences from the job ad; offsets are computed)
# --------------------------------------------------------------------------- #
JOBS = [
    # job_001: matches cand_001 well; Docker missing (preferred only)
    dict(id="job_001", title="Machine Learning Intern", occupation="Machine Learning Intern",
         company="Acme d.o.o.", job_type="internship", city="Zagreb",
         remote=("hybrid", "required", "Hybrid, 2 days in office"),
         skills=[("Python", "required", "Strong Python skills"),
                 ("SQL", "required", "Working knowledge of SQL"),
                 ("Docker", "preferred", "Experience with Docker is considered an advantage.")],
         education=("msc", "Computer Science", "required", "Current MSc student in CS or related"),
         student=(True, "required", "Current MSc student in CS or related"),
         exp=(2, "preferred", "2 years of experience is a plus"),
         languages=[("English", "B2", "required", "Fluent English (B2+)")]),
    # job_002: needs PyTorch + 1 year experience (cand_002 fails required skills)
    dict(id="job_002", title="Junior Data Scientist", occupation="Junior Data Scientist",
         company="DataWorks d.o.o.", job_type="junior", city="Zagreb",
         remote=("onsite", "required", "On-site in Zagreb office"),
         skills=[("Python", "required", "Python for data analysis"),
                 ("SQL", "required", "Solid SQL"),
                 ("PyTorch", "required", "Hands-on PyTorch experience"),
                 ("Docker", "preferred", "Docker is a plus")],
         education=("bsc", "Computer Science", "required",
                    "BSc or MSc in Computer Science, Statistics or similar"),
         student=None,
         exp=(1, "required", "At least 1 year of relevant experience"),
         languages=[("English", "B2", "required", "English B2 or higher")]),
    # job_003: remote backend internship, students only
    dict(id="job_003", title="Backend Developer Intern", occupation="software developer",
         company="CloudNest d.o.o.", job_type="internship", city="Zagreb",
         remote=("remote", "required", "Fully remote position"),
         skills=[("Python", "required", "Python backend development"),
                 ("SQL", "required", "Relational databases and SQL"),
                 ("FastAPI", "preferred", "FastAPI experience is welcome")],
         education=None,
         student=(True, "required", "Open to current university students"),
         exp=None,
         languages=[("English", "B2", "required", "Working English (B2)")]),
    # job_004: onsite in Split; English B2 fails for cand_003 (B1)
    dict(id="job_004", title="Junior Frontend Developer", occupation="frontend developer",
         company="PixelForge d.o.o.", job_type="junior", city="Split",
         remote=("onsite", "required", "Office-based in Split"),
         skills=[("JavaScript", "required", "Solid JavaScript"),
                 ("React", "required", "Production experience with React"),
                 ("TypeScript", "preferred", "TypeScript is a plus")],
         education=("bsc", "Computer Science", "preferred", "BSc preferred"),
         student=None,
         exp=(1, "preferred", "Around 1 year of experience preferred"),
         languages=[("English", "B2", "required", "English B2 required")]),
    # job_005: German B2 required -> language FAIL for everyone without German
    dict(id="job_005", title="Junior Full-stack Developer", occupation="full-stack developer",
         company="Nordic Apps GmbH", job_type="junior", city="Berlin", country="DE",
         remote=("remote", "required", "Remote-first team"),
         skills=[("JavaScript", "required", "JavaScript on the front end"),
                 ("Node.js", "required", "Node.js on the back end"),
                 ("SQL", "required", "SQL databases"),
                 ("Docker", "preferred", "Docker knowledge is an advantage")],
         education=None, student=None, exp=None,
         languages=[("German", "B2", "required", "German B2 required"),
                    ("English", "B2", "required", "English B2 required")]),
    # job_006: Java internship; German only preferred
    dict(id="job_006", title="Java Developer Intern", occupation="java developer",
         company="BankTech d.o.o.", job_type="internship", city="Zagreb",
         remote=("hybrid", "required", "Hybrid, 3 days on-site"),
         skills=[("Java", "required", "Strong Java fundamentals"),
                 ("Spring", "required", "Spring Boot experience"),
                 ("SQL", "preferred", "SQL is a plus")],
         education=("bsc", "Computer Science", "required",
                    "Enrolled in a BSc or MSc programme in computer science"),
         student=(True, "required", "Must be an enrolled student"),
         exp=None,
         languages=[("English", "B1", "required", "Good English (B1)"),
                    ("German", "A2", "preferred", "Basic German is an advantage")]),
    # job_007: DevOps; nobody covers all required skills; hybrid only preferred
    dict(id="job_007", title="Junior DevOps Engineer", occupation="devops engineer",
         company="InfraWorks d.o.o.", job_type="junior", city="Zagreb",
         remote=("hybrid", "preferred", "Hybrid preferred"),
         skills=[("Docker", "required", "Docker in production"),
                 ("Linux", "required", "Linux administration"),
                 ("Kubernetes", "required", "Kubernetes basics"),
                 ("Python", "preferred", "Python scripting is a plus")],
         education=None, student=None,
         exp=(1, "preferred", "1 year of experience preferred"),
         languages=[("English", "B2", "required", "English B2 required")]),
    # job_008: sparse ad: no remote mode, no education, no student requirement
    dict(id="job_008", title="Data Analyst Intern", occupation="data analyst",
         company="RetailCo d.o.o.", job_type="internship", city="Zagreb",
         remote=None,
         skills=[("SQL", "required", "SQL for reporting"),
                 ("Excel", "required", "Advanced Excel"),
                 ("Python", "preferred", "Python is a plus")],
         education=None, student=None, exp=None,
         languages=[("English", "B1", "required", "Good English (B1)")]),
]


def build_job(j) -> NormalizedProfile:
    lines = [j["title"]]
    if j["remote"]:
        lines.append(j["remote"][2])
    lines += [s[2] for s in j["skills"]]
    if j["education"]:
        lines.append(j["education"][3])
    if j["student"]:
        lines.append(j["student"][2])
    if j["exp"]:
        lines.append(j["exp"][2])
    lines += [lang[3] for lang in j["languages"]]
    raw = "\n".join(dict.fromkeys(lines))
    f = "job_text"

    profile = JobProfile(
        job_id=j["id"],
        source=JobSource(dataset="synthetic", raw_text=raw),
        title=j["title"],
        occupation=j["occupation"],
        company=j["company"],
        job_type=j["job_type"],
        location=JobLocation(city=j["city"], country=j.get("country", "HR")),
        remote_mode=(JobRemoteMode(value=j["remote"][0], modality=j["remote"][1],
                                   evidence=ev(raw, j["remote"][2], f)) if j["remote"] else None),
        skills=[JobSkill(name=n, modality=m, evidence=ev(raw, p, f)) for n, m, p in j["skills"]],
        education=(JobEducation(level=j["education"][0], field=j["education"][1],
                                modality=j["education"][2], evidence=ev(raw, j["education"][3], f))
                   if j["education"] else None),
        student_status_required=(JobStudentStatusRequired(
            value=j["student"][0], modality=j["student"][1],
            evidence=ev(raw, j["student"][2], f)) if j["student"] else None),
        experience=(JobExperience(min_years=j["exp"][0], modality=j["exp"][1],
                                  evidence=ev(raw, j["exp"][2], f)) if j["exp"] else None),
        languages=[JobLanguage(language=la, level=lv, modality=m, evidence=ev(raw, p, f))
                   for la, lv, m, p in j["languages"]],
    )
    normalized = Normalized(
        skills=[normalize(s.name, SKILL_MAP, "skill", s.modality, s.evidence)
                for s in profile.skills],
        occupations=[normalize(profile.occupation, OCCUPATION_MAP, "occupation")],
    )
    return NormalizedProfile(entity_type="job", entity_id=j["id"],
                             profile=profile, normalized=normalized)


# --------------------------------------------------------------------------- #
# Retrieval results (hand-set similarities; k = number of mock jobs)
# --------------------------------------------------------------------------- #
SIMILARITY = {
    "cand_001": {"job_001": 0.87, "job_002": 0.84, "job_003": 0.78, "job_008": 0.70,
                 "job_005": 0.62, "job_006": 0.55, "job_004": 0.48, "job_007": 0.41},
    "cand_002": {"job_005": 0.86, "job_004": 0.83, "job_003": 0.72, "job_007": 0.66,
                 "job_008": 0.63, "job_001": 0.52, "job_002": 0.50, "job_006": 0.47},
    "cand_003": {"job_006": 0.88, "job_003": 0.70, "job_005": 0.64, "job_007": 0.58,
                 "job_008": 0.55, "job_002": 0.50, "job_001": 0.46, "job_004": 0.40},
}


def build_retrieval(cand_id: str) -> RetrievalResult:
    ranked = sorted(SIMILARITY[cand_id].items(), key=lambda kv: kv[1], reverse=True)
    return RetrievalResult(
        candidate_id=cand_id,
        encoder_model=ENCODER,
        k=len(ranked),
        results=[RetrievalHit(job_id=jid, semantic_similarity=sim, retrieval_rank=r)
                 for r, (jid, sim) in enumerate(ranked, start=1)],
    )


def dump(name: str, objs) -> None:
    path = OUT_DIR / name
    path.write_text(
        json.dumps([o.model_dump(mode="json") for o in objs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {path} ({len(objs)} objects)")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    dump("candidates.json", [build_candidate(c) for c in CANDIDATES])
    dump("jobs.json", [build_job(j) for j in JOBS])
    dump("retrieval_results.json", [build_retrieval(c["id"]) for c in CANDIDATES])
