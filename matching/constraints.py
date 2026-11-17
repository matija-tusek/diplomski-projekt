"""Hard constraints: requirements the job marks as `required`.

A required requirement gives PASS / FAIL / UNKNOWN (CV does not say), and required = True.
Anything else (preferred, unspecified or not stated) gives NOT_APPLICABLE and required = False;
the score already rewards or penalizes it. Only a FAIL excludes a job, never an UNKNOWN.
"""
import schemas.models as models
from matching.features import language_check

REQUIRED = models.Modality.REQUIRED
Status = models.ConstraintStatus


def _check(name: str, hard: bool, ok: bool | None, **value) -> models.ConstraintCheck:
    if not hard:
        status = Status.NOT_APPLICABLE
    else:
        status = Status.UNKNOWN if ok is None else Status.PASS if ok else Status.FAIL
    return models.ConstraintCheck(constraint=name, status=status, required=hard, value=value)


def _hard(requirement) -> bool:
    return requirement is not None and requirement.modality == REQUIRED


def check_constraints(cand: models.NormalizedProfile, job: models.NormalizedProfile,
                      f: models.MatchFeatures) -> models.Constraints:
    j = job.profile
    student, edu, exp, remote = j.student_status_required, j.education, j.experience, j.remote_mode
    is_student = cand.profile.student_status.is_student
    required_languages = [lang for lang in j.languages if lang.modality == REQUIRED]
    language_share, unmet_languages = language_check(cand, required_languages)

    checks = [
        _check("required_skills", any(s.modality == REQUIRED for s in job.normalized.skills),
               not f.missing_required_skills,
               missing_required_skills=f.missing_required_skills),
        _check("student_status", _hard(student),
               None if student is None or is_student is None else is_student == student.value,
               is_student=is_student, required_value=student.value if student else None),
        _check("education", _hard(edu),
               None if f.education_match is None else f.education_match == 1.0,
               education_match=f.education_match,
               required_level=edu.level.value if edu and edu.level else None),
        _check("experience", _hard(exp),
               None if f.experience_gap is None else f.experience_gap == 0,
               experience_gap=f.experience_gap, min_years=exp.min_years if exp else None),
        _check("language", bool(required_languages),
               False if unmet_languages else None if language_share is None else True,
               language_match=f.language_match, missing_languages=unmet_languages),
        # One step apart (hybrid vs. remote) passes; only opposite ends (onsite vs. remote) fail.
        _check("remote_mode", _hard(remote),
               None if f.remote_preference_match is None else f.remote_preference_match > 0,
               remote_preference_match=f.remote_preference_match,
               required_mode=remote.value.value if remote else None),
    ]
    hard = [c.status for c in checks if c.required]
    overall = Status.FAIL if Status.FAIL in hard else Status.UNKNOWN if Status.UNKNOWN in hard else Status.PASS
    return models.Constraints(overall_status=overall, checks=checks)


_DETAIL = {
    "required_skills": lambda v: "missing " + ", ".join(v["missing_required_skills"]),
    "student_status": lambda v: "students only" if v["required_value"] else "not for students",
    "education": lambda v: f"{v['required_level']} required",
    "experience": lambda v: f"{v['min_years']:g}+ years required",
    "language": lambda v: ", ".join(v["missing_languages"]) + " required",
    "remote_mode": lambda v: f"{v['required_mode']} required",
}


def exclusion_reason(constraints: models.Constraints) -> str:
    """e.g. 'constraint FAIL: language (German B2 required)'"""
    failed = [c for c in constraints.checks if c.status == Status.FAIL]
    return "constraint FAIL: " + "; ".join(f"{c.constraint} ({_DETAIL[c.constraint](c.value)})" for c in failed)
