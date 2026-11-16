"""Učitavanje NormalizedProfile objekata iz datoteka."""

import json
from pathlib import Path


def load_profiles(path, entity_type=None):
    """Prima .json (jedan objekt ili lista), .jsonl ili mapu s .json datotekama."""
    path = Path(path)
    if path.is_dir():
        profiles = [p for f in sorted(path.glob("*.json")) for p in load_profiles(f)]
    elif path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        profiles = [json.loads(line) for line in lines if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = data if isinstance(data, list) else [data]

    if entity_type:
        wrong = [p.get("entity_id") for p in profiles if p.get("entity_type") != entity_type]
        if wrong:
            raise ValueError(f"očekivan entity_type '{entity_type}', ali ga nemaju: {wrong}")
    return profiles


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
