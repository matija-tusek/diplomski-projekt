from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


def clean_term(value: str) -> str:
    return " ".join(value.casefold().split())


def _labels(raw: str) -> list[str]:
    return [label.strip() for label in re.split(r"[\r\n|]+", raw or "") if label.strip()]


def load_esco_index(path: str | Path) -> dict[str, list[tuple[dict[str, str], str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ESCO CSV not found: {path}")

    index: dict[str, list[tuple[dict[str, str], str]]] = {}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
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
            for term, method in candidates:
                key = clean_term(term)
                existing = index.setdefault(key, [])
                if not any(record[0]["uri"] == uri for record in existing):
                    existing.append((concept, method))
    print(f"ESCO: loaded {count} concepts from {path.name} ({len(index)} terms in index)")
    return index


def map_term(term: str, index: dict[str, list[tuple[dict[str, str], str]]]) -> dict[str, Any]:
    options = index.get(clean_term(term), [])
    if len(options) == 1:
        concept, method = options[0]
        return {
            "source_term": term,
            "normalized_label": concept["preferred_label"],
            "esco_concept": concept,
            "mapping": {"status": "EXACT", "method": method, "confidence": 1.0},
        }
    return {
        "source_term": term,
        "normalized_label": term,
        "esco_concept": None,
        "mapping": {"status": "UNMAPPED", "method": "none", "confidence": 0.0},
    }
