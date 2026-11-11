"""Load raw documents (CVs or job ads) from a file or folder.

Supported inputs:
- a single .txt file                      -> one document, id = file name without extension
- a folder of .txt files                  -> one document per file
- .csv / .json (list of objects) / .jsonl -> one document per row; pick the columns with
  `text_column` and `id_column` (row number if no id column)

Datasets are never committed (see the root .gitignore); put them under data/.
"""
import csv
import json
import sys
from pathlib import Path


def _rows(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # job ads can be long
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        yield from json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .txt, .csv, .json or .jsonl)")


def load_documents(path, text_column="text", id_column=None, limit=None):
    """Yield (doc_id, text) pairs."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        docs = ((f.stem, f.read_text(encoding="utf-8")) for f in files)
    elif path.suffix.lower() == ".txt":
        docs = iter([(path.stem, path.read_text(encoding="utf-8"))])
    else:
        docs = _table_documents(path, text_column, id_column)

    for i, (doc_id, text) in enumerate(docs):
        if limit is not None and i >= limit:
            break
        yield doc_id, text


def _table_documents(path, text_column, id_column):
    for i, row in enumerate(_rows(path)):
        if text_column not in row:
            raise KeyError(f"Column '{text_column}' not found. Available: {', '.join(row)}")
        text = row[text_column]
        if not text or not str(text).strip():
            continue
        doc_id = str(row[id_column]) if id_column else str(i)
        yield doc_id, str(text)
