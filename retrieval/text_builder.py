"""Iz NormalizedProfile objekata slaže tekst koji ide u encoder.

U tekst ulazi samo ono što opisuje sadržaj posla: uloga, vještine, iskustvo
i obrazovanje. Lokacija, remote mode i jezici namjerno su izostavljeni, jer
su to tvrdi uvjeti koje provjerava Grupa 4.
"""


def _dedupe(items):
    """Uklanja prazne i duple vrijednosti, čuva redoslijed."""
    seen, out = set(), []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def esco_label(entry, use_esco=True):
    """ESCO preferred_label ako postoji, inače izvorni pojam (UNMAPPED)."""
    concept = entry.get("esco_concept")
    if use_esco and concept and concept.get("preferred_label"):
        return concept["preferred_label"]
    return entry.get("source_term")


def _labels(profile, key, use_esco):
    entries = (profile.get("normalized") or {}).get(key) or []
    return _dedupe(esco_label(e, use_esco) for e in entries)


def _education(edu):
    if not edu:
        return None
    return " ".join(x for x in (edu.get("level"), edu.get("field")) if x) or None


def candidate_text(candidate, use_esco=True):
    p = candidate["profile"]
    prefs = p.get("preferences") or {}

    roles = _dedupe((prefs.get("target_roles") or []) + _labels(candidate, "occupations", use_esco))
    skills = _labels(candidate, "skills", use_esco)
    experience = _dedupe(
        ": ".join(x for x in (e.get("title"), e.get("description")) if x)
        for e in p.get("experience") or []
    )
    education = _dedupe(_education(e) for e in p.get("education") or [])
    interests = _dedupe(prefs.get("interests") or [])

    lines = [
        ("Target roles", ", ".join(roles)),
        ("Skills", ", ".join(skills)),
        ("Experience", "; ".join(experience)),
        ("Education", ", ".join(education)),
        ("Interests", ", ".join(interests)),
    ]
    return "\n".join(f"{name}: {value}." for name, value in lines if value)


def job_text(job, use_esco=True):
    p = job["profile"]

    occupations = [o for o in _labels(job, "occupations", use_esco)
                   if o.lower() != (p.get("title") or "").lower()]
    skills = _labels(job, "skills", use_esco)
    education = _education(p.get("education"))
    min_years = (p.get("experience") or {}).get("min_years")

    lines = [
        ("Role", p.get("title")),
        ("Occupation", ", ".join(occupations)),
        ("Skills", ", ".join(skills)),
        ("Education", education),
        ("Experience", f"{min_years} years" if min_years else None),
    ]
    return "\n".join(f"{name}: {value}." for name, value in lines if value)
