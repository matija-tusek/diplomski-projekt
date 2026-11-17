import argparse
import copy
import json
from pathlib import Path

from esco_embeddings import load_embeddings
from esco_loader import load_esco_index, map_term


def normalize_profile(profile, skills_index, occupations_index, skills_embeddings=None, occupations_embeddings=None):
    if profile.get("schema_version") != "1.0":
        raise ValueError("Podrzana je samo verzija JSON sheme 1.0")

    # Determine if input is a candidate or a job profile.
    if "candidate_id" in profile and "job_id" not in profile:
        entity_type = "candidate"
        entity_id = profile["candidate_id"]
        # Grupa 1 stavlja null kad podatak ne postoji, zato "or"
        occupations = (profile.get("preferences") or {}).get("target_roles") or []
    elif "job_id" in profile and "candidate_id" not in profile:
        entity_type = "job"
        entity_id = profile["job_id"]
        occupations = [profile["occupation"]] if profile.get("occupation") else []
    else:
        raise ValueError("Ocekujem CandidateProfile ili JobProfile")

    # Normalize skills.
    normalized_skills = []

    for skill in profile.get("skills", []):
        name = skill.get("name")
        if not name:
            raise ValueError("Skill has no name")

        result = map_term(name, skills_index, embeddings=skills_embeddings)
        result["modality"] = skill.get("modality", "unspecified")
        result["evidence"] = copy.deepcopy(skill.get("evidence"))
        normalized_skills.append(result)

    # 2. Normalize occupations.
    normalized_occupations = []

    for occupation in occupations:
        result = map_term(occupation, occupations_index, is_occupation=True, embeddings=occupations_embeddings)
        normalized_occupations.append(result)
            
    # 3. Assemble the normalized profile JSON.
    return {
        "schema_version": "1.0",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "profile": copy.deepcopy(profile),
        "normalized": {
            "skills": normalized_skills,
            "occupations": normalized_occupations,
        },
    }


def load_profiles(path):
    # .json (jedan profil ili lista), .jsonl (profil po retku, tako Grupa 1 sprema vise CV-ova)
    # ili mapa s .json datotekama
    if path.is_dir():
        profiles = []
        for file in sorted(path.glob("*.json")):
            profiles += load_profiles(file)
        return profiles

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    return [data]


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        lines = [json.dumps(result, ensure_ascii=False) for result in results]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif len(results) == 1:
        # jedan profil se sprema kao objekt, isto kao prije
        path.write_text(json.dumps(results[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    # Arguments from the terminal: paths for input JSON, ESCO CSV, and output JSON.
    parser = argparse.ArgumentParser(description="ESCO normalization of skills and occupations")
    parser.add_argument("--input", required=True, type=Path, help=".json, .jsonl or a folder of .json files")
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--occupations", type=Path)
    parser.add_argument("--output", required=True, type=Path, help=".json or .jsonl")
    parser.add_argument("--no-embeddings", action="store_true", help="only exact + fuzzy, faster")
    args = parser.parse_args()

    # Load profile and ESCO data.
    profiles = load_profiles(args.input)
    skills_index = load_esco_index(args.skills)
    occupations_index = {}
    if args.occupations:
        occupations_index = load_esco_index(args.occupations)

    # prvi put se vektori racunaju par minuta, poslije se samo ucitaju s diska
    skills_embeddings = None
    occupations_embeddings = None
    if not args.no_embeddings:
        skills_embeddings = load_embeddings(args.skills)
        if args.occupations:
            occupations_embeddings = load_embeddings(args.occupations)

    # Normalize and save the result.
    # ESCO i model su ucitani samo jednom, bez obzira koliko je profila
    results = []
    for profile in profiles:
        results.append(normalize_profile(profile, skills_index, occupations_index, skills_embeddings, occupations_embeddings))
    save_results(results, args.output)

    # statistika za sve profile zajedno
    all_skills = [skill for result in results for skill in result["normalized"]["skills"]]
    all_occupations = [occupation for result in results for occupation in result["normalized"]["occupations"]]

    # Short statistics of mapping success.
    counts = {"EXACT": 0, "RELATED": 0, "UNMAPPED": 0}
    for skill in all_skills:
        status = skill["mapping"]["status"]
        counts[status] += 1

    if len(results) == 1:
        print(f"Normalized {results[0]['entity_type']} {results[0]['entity_id']}: {counts}")
    else:
        print(f"Normalized {len(results)} profiles, skills: {counts}")

    occupation_counts = {"EXACT": 0, "RELATED": 0, "UNMAPPED": 0}
    for occupation in all_occupations:
        occupation_counts[occupation["mapping"]["status"]] += 1
    print(f"Occupations: {occupation_counts}")

    # koliko je pojmova pronadeno kojom metodom (exact_lookup, alias_lookup, none...)
    methods = {}
    for item in all_skills + all_occupations:
        method = item["mapping"]["method"]
        methods[method] = methods.get(method, 0) + 1
    print(f"Methods: {methods}")

    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
