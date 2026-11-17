from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from esco_embeddings import EMBEDDING_CUTOFF, embedding_lookup


def clean_term(value: str) -> str:
    return " ".join(value.casefold().split())


def _labels(raw: str) -> list[str]:
    return [label.strip() for label in re.split(r"[\r\n|]+", raw or "") if label.strip()]


# rijeci koje kod zanimanja samo govore o razini, a ne o samom poslu
SENIORITY_WORDS = ["intern", "internship", "junior", "senior", "trainee", "student",
                   "praktikant", "pripravnik", "jr", "sr", "(m/f)", "(f/m)", "(m/ž)"]

# najcesci padezni nastavci (Pythona, Pythonom, Javi, Excelu...)
CROATIAN_ENDINGS = ["ima", "om", "em", "a", "e", "i", "u"]


def remove_seniority(text):
    # "Junior Data Scientist" -> "Data Scientist"
    words = [w for w in text.split() if w.strip(".").casefold() not in SENIORITY_WORDS]
    return " ".join(words)


def term_variants(term, is_occupation=False):
    variants = [clean_term(term)]

    # machine-learning -> machine learning, "Python." -> python
    no_dashes = variants[0].replace("-", " ").replace("_", " ").strip(" .,;:")
    variants.append(" ".join(no_dashes.split()))

    # SQL-a, SQL-om -> sql
    if "-" in variants[0]:
        word, ending = variants[0].rsplit("-", 1)
        if len(ending) <= 3:
            variants.append(word)

    # Junior Data Scientist -> data scientist
    if is_occupation:
        for variant in list(variants):
            variants.append(remove_seniority(variant))

    # pythona -> python, javi -> java
    # ovo se ionako proba tek kad originalni oblik nije nadjen
    for variant in list(variants):
        for ending in CROATIAN_ENDINGS:
            base = variant[:-len(ending)]
            if variant.endswith(ending) and len(base) >= 3:
                variants.append(base)
                variants.append(base + "a")

    # bez duplikata i praznih, redoslijed ostaje isti
    result = []
    for variant in variants:
        if variant and variant not in result:
            result.append(variant)
    return result


def load_esco_index(path: str | Path) -> dict[str, list[tuple[dict[str, str], str, str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ESCO CSV not found: {path}")

    index: dict[str, list[tuple[dict[str, str], str, str]]] = {}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        # ESCO je obican CSV sa zarezima, Sniffer je krivo pogadao format i stvarao lazni koncept
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {"conceptUri", "preferredLabel"}
        if not required.issubset(columns):
            raise ValueError(
                f"ESCO CSV needs to have columns {sorted(required)}, got: {sorted(columns)}"
            )
        for row in reader:
            uri = (row.get("conceptUri") or "").strip()
            label = (row.get("preferredLabel") or "").strip()
            if not uri or not label:
                continue
            concept = {"uri": uri, "preferred_label": label}
            count += 1
            candidates = [(label, "exact_lookup")]
            candidates += [(alias, "alias_lookup") for alias in _labels(row.get("altLabels") or "")]
            candidates += [(alias, "alias_lookup") for alias in _labels(row.get("hiddenLabels") or "")]
            for term, method in candidates:
                key = clean_term(term)
                existing = index.setdefault(key, [])
                if not any(record[0]["uri"] == uri for record in existing):
                    existing.append((concept, method, term))  # term = naziv kako pise u ESCO-u
    print(f"ESCO: loaded {count} concepts from {path.name} ({len(index)} terms in index)")
    return index


def pick_one(options):
    # vise pogodaka (npr. "SQL"), a samo jedan je sluzbeni naziv -> uzmi njega
    if len(options) > 1:
        official = [option for option in options if option[1] == "exact_lookup"]
        if len(official) == 1:
            options = official

    if len(options) == 1:
        return options[0]
    return None  # nema pogotka ili je i dalje dvosmisleno


# "Pyhton" vs "python" ima tocno 83.3, zato 83
FUZZY_CUTOFF = 83
# zanimanja su duze fraze (developer, engineer...) pa par slova vec mijenja posao,
# npr. "ML engineer" je s 83 postao "metal engineer"
FUZZY_CUTOFF_OCCUPATION = 92


def fuzzy_lookup(term, index, is_occupation=False):
    best = None  # (concept, esco_label, score)
    cutoff = FUZZY_CUTOFF_OCCUPATION if is_occupation else FUZZY_CUTOFF

    for key in term_variants(term, is_occupation):
        # na kratkim pojmovima (C, R, Go) bi se svasta poklopilo
        if len(key) < 4:
            continue

        # index.keys() se ne kopira, rapidfuzz samo prodje kroz njih
        hits = process.extract(key, index.keys(), scorer=fuzz.ratio, score_cutoff=cutoff, limit=5)

        for esco_key, score, _ in hits:
            # tipfeler je rijetko u prvom slovu, a ovo odbaci stvari tipa html -> phtml
            if esco_key[0] != key[0]:
                continue
            option = pick_one(index[esco_key])
            if option is None:
                continue
            if best is None or score > best[2]:
                best = (option[0], option[2], score)
            break  # hits su vec sortirani, prvi dobar je i najbolji

    return best


def map_term(
    term: str,
    index: dict[str, list[tuple[dict[str, str], str, str]]],
    is_occupation: bool = False,
    embeddings: dict | None = None,
) -> dict[str, Any]:
    # probaj redom: originalni oblik, pa bez crtica, pa bez padeza...
    for key in term_variants(term, is_occupation):
        option = pick_one(index.get(key, []))
        if option:
            concept, method, esco_label = option
            return {
                "source_term": term,
                "normalized_label": esco_label,
                "esco_concept": concept,
                "mapping": {"status": "EXACT", "method": method, "confidence": 1.0},
            }

    # doslovno nije nadjeno, mozda je tipfeler ("Pyhton")
    fuzzy = fuzzy_lookup(term, index, is_occupation)
    if fuzzy:
        concept, esco_label, score = fuzzy
        return {
            "source_term": term,
            "normalized_label": esco_label,
            "esco_concept": concept,
            "mapping": {"status": "RELATED", "method": "fuzzy_match", "confidence": round(score / 100, 2)},
        }

    # ni fuzzy nije pomogao, trazimo po znacenju (npr. "Excel" -> use spreadsheets software)
    if embeddings is not None:
        query = remove_seniority(term) if is_occupation else term
        concept, esco_label, score = embedding_lookup(query, embeddings)[0]

        # isti naziv zna imati vise koncepata ("data scientist"), pick_one uzme sluzbeni
        option = pick_one(index.get(clean_term(esco_label), []))
        if option:
            concept = option[0]

        if score >= EMBEDDING_CUTOFF:
            return {
                "source_term": term,
                "normalized_label": term,  # kao u primjeru iz Ideje: FastAPI ostaje FastAPI
                "esco_concept": concept,
                "mapping": {"status": "RELATED", "method": "embedding_similarity", "confidence": round(score, 2)},
            }

    return {
        "source_term": term,
        "normalized_label": term,
        "esco_concept": None,
        "mapping": {"status": "UNMAPPED", "method": "none", "confidence": 0.0},
    }
