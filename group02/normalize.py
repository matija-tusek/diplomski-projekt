import argparse
import copy
import json
from pathlib import Path

from esco_loader import load_esco_index, map_term


def normalize_profile(profile, skills_index, occupations_index):
    if profile.get("schema_version") != "1.0":
        raise ValueError("Podrzana je samo verzija JSON sheme 1.0")

    # Determine if input is a candidate or a job profile.
    if "candidate_id" in profile and "job_id" not in profile:
        entity_type = "candidate"
        entity_id = profile["candidate_id"]
        occupations = profile.get("preferences", {}).get("target_roles", [])
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

        result = map_term(name, skills_index)
        result["modality"] = skill.get("modality", "unspecified")
        result["evidence"] = copy.deepcopy(skill.get("evidence"))
        normalized_skills.append(result)

    # 2. Normalize occupations.
    normalized_occupations = []

    for occupation in occupations:
        result = map_term(occupation, occupations_index)
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


def main():
    # Arguments from the terminal: paths for input JSON, ESCO CSV, and output JSON.
    parser = argparse.ArgumentParser(description="ESCO normalization of skills and occupations")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--occupations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Load profile and ESCO data.
    profile = json.loads(args.input.read_text(encoding="utf-8"))
    skills_index = load_esco_index(args.skills)
    occupations_index = {}
    if args.occupations:
        occupations_index = load_esco_index(args.occupations)

    # Normalize and save the result.
    result = normalize_profile(profile, skills_index, occupations_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Short statistics of mapping success.
    counts = {"EXACT": 0, "RELATED": 0, "UNMAPPED": 0}
    for skill in result["normalized"]["skills"]:
        status = skill["mapping"]["status"]
        counts[status] += 1

    print(f"Normalized {result['entity_type']} {result['entity_id']}: {counts}")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
