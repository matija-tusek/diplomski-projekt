import schemas.models as models

# Lowest to highest; a level's rank is its list index.
EDUCATION_ORDER = list(models.EducationLevel)  # high_school < bsc < msc < phd
LANGUAGE_ORDER = list(models.LanguageLevel)    # A1 < ... < C2 < native
REMOTE_ORDER = [models.RemoteMode.ONSITE, models.RemoteMode.HYBRID, models.RemoteMode.REMOTE]

# Hand-set weights. Education and language are left out: they are pass/fail constraints, so
# every eligible job scores the same on them. Change a weight -> bump the version.
SCORE_WEIGHTS_VERSION = "w_v1"
SCORE_WEIGHTS = {
    "semantic_similarity": 0.30,
    "required_skill_coverage": 0.15,
    "preferred_skill_coverage": 0.15,
    "esco_occupation_similarity": 0.15,
    "experience_fit": 0.10,  # derived from experience_gap, see calculate_score
    "seniority_match": 0.05,
    "remote_preference_match": 0.05,
    "location_match": 0.05,
}


def calculate_score(features: models.MatchFeatures) -> float:
    """Weighted average of the known features. A None (unknown) feature is skipped and the
    other weights are rescaled, so missing information neither helps nor hurts."""
    # LIMITATION: preferred experience is penalized like required (features carry no modality).
    # Upgrade: fit weights with logistic regression on labeled pairs -> w_v2.
    values = features.model_dump()
    gap = values["experience_gap"]
    values["experience_fit"] = None if gap is None else 1 / (1 + gap)  # 0 y -> 1.0, 1 y -> 0.5
    known = {k: w for k, w in SCORE_WEIGHTS.items() if values[k] is not None}
    score = sum(w * values[k] for k, w in known.items()) / sum(known.values())
    return round(min(max(score, 0.0), 1.0), 3)  # cosine similarity can be negative

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

def _skill_keys(skill: models.NormalizedTerm) -> set[str]:
    """Keys under which two skills count as the same: the name as written, the normalized
    label and, only for EXACT mappings, the ESCO URI. A RELATED URI is a broader concept
    (FastAPI -> "web programming", same as Django), so matching on it would be a false hit."""
    keys = {skill.source_term.casefold(), skill.normalized_label.casefold()}
    if skill.esco_concept and skill.mapping.status == models.MappingStatus.EXACT:
        keys.add(skill.esco_concept.uri)
    return keys


def skill_coverage(cand: models.NormalizedProfile, job: models.NormalizedProfile, modality: str) -> tuple[float, list[str]]:
    """Share of the job's skills with this modality that the candidate has.
    Returns (coverage, missing), missing as the job wrote them.
    Job lists no such skills -> (1.0, []): nothing is asked, so nothing is missing."""
    cand_keys = set().union(*(_skill_keys(s) for s in cand.normalized.skills))

    # The same skill can appear twice in an ad; count it once.
    wanted = {}
    for s in job.normalized.skills:
        if s.modality == modality:
            wanted.setdefault(s.source_term.casefold(), s)
    if not wanted:
        return 1.0, []

    missing = [s.source_term for s in wanted.values() if not (_skill_keys(s) & cand_keys)]
    return round(1 - len(missing) / len(wanted), 2), missing


def occupation_similarity(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Closeness of the candidate's target roles to the job's occupation.
    Same term -> 1.0. Same ESCO concept -> product of both mapping confidences (how sure we
    are that both really are that concept). Nothing shared -> 0.0. Either side has no
    occupation -> None."""
    # LIMITATION: only exact concept overlap; "data scientist" vs "data analyst" scores 0.
    # Upgrade: ESCO occupation hierarchy (ISCO groups) or embedding similarity from Group 3.
    if not cand.normalized.occupations or not job.normalized.occupations:
        return None
    best = 0.0
    for c in cand.normalized.occupations:
        for j in job.normalized.occupations:
            if c.source_term.casefold() == j.source_term.casefold():
                return 1.0
            if c.esco_concept and j.esco_concept and c.esco_concept.uri == j.esco_concept.uri:
                best = max(best, c.mapping.confidence * j.mapping.confidence)
    return round(best, 2)


def education_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """1.0 if the candidate's highest level (enrolled counts, dropped out does not) is at
    least the job's level, else 0.0. Job states no education -> 1.0. Level unknown on
    either side -> None."""
    # LIMITATION: field of study is not compared ("CS or related" is too fuzzy for string matching).
    req = job.profile.education
    if req is None:
        return 1.0
    if req.level is None:
        return None
    levels = [e.level for e in cand.profile.education
              if e.level is not None and e.status != models.EducationStatus.DROPPED_OUT]
    if not levels:
        return None
    best = max(EDUCATION_ORDER.index(lv) for lv in levels)
    return 1.0 if best >= EDUCATION_ORDER.index(req.level) else 0.0


def seniority_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """1.0 if the job's job_type is one the candidate is looking for, else 0.0.
    Candidate states no job types -> None."""
    prefs = cand.profile.preferences
    if not prefs or not prefs.job_types:
        return None
    return 1.0 if job.profile.job_type in prefs.job_types else 0.0


def experience_gap(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Years the candidate is short of the job's min_years; 0.0 = meets it or no requirement.
    An entry with unknown months (not the same as 0 months) makes the gap None, unless
    the known months already cover the requirement."""
    # LIMITATION: months are summed, so overlapping jobs count twice. Upgrade: merge date ranges.
    req = job.profile.experience
    if req is None:
        return 0.0
    months = [e.months for e in cand.profile.experience]
    gap = max(0.0, req.min_years - sum(m for m in months if m is not None) / 12)
    if gap > 0 and None in months:
        return None
    return round(gap, 2)


def language_check(cand: models.NormalizedProfile, asked: list[models.JobLanguage]) -> tuple[float | None, list[str]]:
    """Share of the `asked` languages the candidate speaks at the asked level or higher, plus
    the unmet ones as text ("German B2"). Share is None when a level the comparison needs is
    unknown on the candidate's side; the unmet list is still filled. Nothing asked -> (1.0, [])."""
    # LIMITATION: names compared as written ("German" != "njemački"), and a language missing
    # from the CV counts as not spoken. Upgrade: normalize language names to ISO 639 codes.
    if not asked:
        return 1.0, []
    have = {lang.language.casefold(): lang.level for lang in cand.profile.languages}
    unmet, unknown = [], False
    for lang in asked:
        key = lang.language.casefold()
        text = f"{lang.language} {lang.level.value}" if lang.level else lang.language
        if key not in have:
            unmet.append(text)
        elif lang.level is None:  # job names the language but no level: speaking it is enough
            pass
        elif have[key] is None:
            unknown = True
        elif LANGUAGE_ORDER.index(have[key]) < LANGUAGE_ORDER.index(lang.level):
            unmet.append(text)
    if unknown:
        return None, unmet
    return round(1 - len(unmet) / len(asked), 2), unmet


def language_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """language_check over the job's required and unspecified languages (preferred ones are
    only a plus). Job asks for none -> 1.0."""
    asked = [lang for lang in job.profile.languages if lang.modality != models.Modality.PREFERRED]
    return language_check(cand, asked)[0]


def location_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """1.0 if the job's city or country is one of the candidate's preferred locations, else
    0.0. Fully remote job -> 1.0, location does not matter. No location on either side -> None."""
    # LIMITATION: exact names, so "Croatia" does not match country "HR". Upgrade: country lookup table.
    remote = job.profile.remote_mode
    if remote and remote.value == models.RemoteMode.REMOTE:
        return 1.0
    prefs = cand.profile.preferences
    loc = job.profile.location
    wanted = {p.casefold() for p in prefs.locations} if prefs and prefs.locations else set()
    place = {p.casefold() for p in (loc.city, loc.country) if p} if loc else set()
    if not wanted or not place:
        return None
    return 1.0 if wanted & place else 0.0


def remote_preference_match(cand: models.NormalizedProfile, job: models.NormalizedProfile) -> float | None:
    """Candidate remote_mode vs. job remote_mode on the scale onsite - hybrid - remote:
    same 1.0, one step apart 0.5, opposite ends 0.0. 'any' on either side fits everything.
    Either side unknown -> None."""
    prefs = cand.profile.preferences
    want = prefs.remote_mode if prefs else None
    offered = job.profile.remote_mode.value if job.profile.remote_mode else None
    if want is None or offered is None:
        return None
    if models.RemoteMode.ANY in (want, offered):
        return 1.0
    return 1 - abs(REMOTE_ORDER.index(want) - REMOTE_ORDER.index(offered)) / 2