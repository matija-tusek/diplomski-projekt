# Group 1: LLM extraction (CV / job ad → CandidateProfile / JobProfile)

Owners: Matija Tušek, Antonio Valec. Pipeline stage 1 of 4.

This module turns raw text into the shared JSON contract **schema v1.0**
(`schemas/project_schemas.json`). A CV becomes a `CandidateProfile` and a job ad a `JobProfile`.
Group 2 consumes both (skill normalization + ESCO).

```
CV text  ──► Qwen3-4B (Ollama, structured output) ──► locate evidence ──► validate ──► CandidateProfile
Job text ──►               "                      ──►        "        ──►    "     ──► JobProfile
```

The LLM is only a translator from text to JSON. It decides nothing that code can check or compute.

## Quick start (Windows, from the repo root)

```powershell
# 1. once: Ollama + model (~2.5 GB)
winget install --id Ollama.Ollama -e
ollama pull qwen3:4b-instruct

# 2. once: Python environment (shared root requirements.txt)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. extract one document
.\.venv\Scripts\python.exe -m extraction.cli cv  extraction/gold/cv/cv_01.txt
.\.venv\Scripts\python.exe -m extraction.cli job extraction/gold/job/job_01.txt
```

Output: `output/cand_cv_01.json`, `output/job_job_01.json`. Each run also prints how many evidence
quotes were found in the source and lists any item that was dropped.

### Datasets

Datasets are never committed (`data/` is git-ignored). Put the file under `data/` and point the CLI
at it, naming the text column and (optionally) the id column:

```powershell
.\.venv\Scripts\python.exe -m extraction.cli job data/jobs.csv --text-column description --id-column id --limit 20
.\.venv\Scripts\python.exe -m extraction.cli cv  data/cvs.jsonl --text-column resume_text --limit 20
.\.venv\Scripts\python.exe -m extraction.cli cv  data/cv_folder/          # one .txt per CV
```

Supported: a `.txt` file, a folder of `.txt` files, `.csv`, `.json` (list of objects), `.jsonl`.
Several documents are written to `output/<cv|job>_profiles.jsonl`, one profile per line.

### Using it from Python (e.g. Group 4's pipeline)

```python
from extraction.extractor import extract_candidate, extract_job

profile, report = extract_candidate(cv_text, "cand_017")   # profile: dict, CandidateProfile v1.0
job, report = extract_job(ad_text, "job_284", dataset="synthetic")
```

## Files

| File | What it does |
|------|--------------|
| `schemas.py` | `Llm*` models (what the LLM fills in) and the v1.0 contract models (final output is validated against them). |
| `prompts.py` | System prompts: rules for each field, modality wording, "never guess". |
| `evidence.py` | Finds each evidence quote in the source and computes `char_start` / `char_end`. |
| `extractor.py` | `extract_candidate()` / `extract_job()`: LLM call → evidence → derived values → validation. |
| `datasets.py` | Loads documents from `.txt`, folders, `.csv`, `.json`, `.jsonl`. |
| `cli.py` | Command line for single files and whole datasets. |
| `evaluate.py` | Scores extraction against the hand-labelled gold set. |
| `gold/` | 5 CVs + 5 job ads (English and Croatian) with `labels.json`. |
| `test_extraction.py` | Offline unit tests (LLM mocked). |

## How it works

1. **Structured output.** The Pydantic `Llm*` models are passed to Ollama as `format`
   (JSON schema) with `temperature=0`. The model can only produce JSON in that shape, and enums are
   `Literal[...]`, so an invalid value (e.g. `"B2+"`, `"must have"`) cannot be generated.
2. **Evidence first.** In every item, `evidence` is the first property, so the model copies the
   source phrase before writing the value read from it. Skills are asked as groups
   `{evidence, names: [...]}`, because one line often lists several skills ("pandas and
   scikit-learn"); code expands each group into one item per skill. For CVs, every experience entry
   also lists `skills_used`, grounded on its description sentence, so skills that appear only in
   experience ("lab sessions in C") are not lost.
3. **Offsets are computed, not generated.** LLMs cannot count characters. `evidence.py` searches
   for the quote: exact first, then ignoring case, spaces and punctuation (catches slips like
   `German (A-2)` vs. `German (A2)`). The stored evidence is always the real slice of the source:
   `raw_text[char_start:char_end] == evidence.text`.
4. **Ungrounded items are dropped.** If a quote is not in the source at all, the model paraphrased
   or invented it, and the item is removed and reported. Every item in the output therefore has
   valid, non-null evidence.
5. **Values must be written in their evidence.** A CEFR level is kept only if the quote contains it
   (`"fluent English"` → level `null`, not C1); `native` needs "native" / "mother tongue" /
   "materinski"; an education level needs a degree word ("MSc", "bachelor", "diplomski"; being a
   "student" is not a degree); `experience.min_years` needs a number or a negation in its quote.
   Otherwise the value becomes `null`.
6. **Code derives what the LLM is bad at.** The model returns start/end dates of each job, and
   `extractor.py` computes `months` (a year-only range like "2024–2025" gives `null`, since it could
   be 2 or 24 months). Duplicate skills are merged, country codes are checked to be 2 letters, and an
   empty list for a preference becomes `null` ("not stated").
7. **Contract validation.** The result is validated with the v1.0 contract models
   (`extra="forbid"`, enums, offsets ≥ 0) before it is returned.

### Rules the output follows

- **Unknown is not false.** Anything not stated is `null`: `is_student`, CEFR level ("fluent" is
  not a level), `months`, preferences, and whole job requirements (`education`, `experience`,
  `student_status_required`, `remote_mode` are `null` objects when the ad says nothing).
- **Modality** for every job requirement: `required` (must, strong, required, or listed as a
  requirement), `preferred` (a plus, an advantage, nice to have), `unspecified` (just mentioned,
  e.g. "our stack: Python, Airflow").
- **IDs**: `cand_<id>` / `job_<id>`, where `<id>` is the file name or the dataset id column.
- `source.raw_text` holds the full original text, so every offset can be checked.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m extraction.evaluate          # runs the LLM on all 10 gold docs
.\.venv\Scripts\python.exe -m extraction.evaluate --reuse  # re-score cached predictions only
.\.venv\Scripts\python.exe -m unittest extraction.test_extraction
```

Precision / recall / F1 for list fields (skills, languages, education, experience), accuracy for
attributes (modality, CEFR level, months, ...) and single values, where `null` has to be
predicted as `null`. Every error is listed, and the report is saved to `output/eval_report.json`.

### Results (qwen3:4b-instruct, RTX 3050 4 GB, 2026-11-11)

| List field | Precision | Recall | F1 |
|------------|-----------|--------|----|
| cv.skills | 0.97 | 0.95 | 0.96 |
| cv.education / languages / experience | 1.00 | 1.00 | 1.00 |
| job.skills | 1.00 | 1.00 | 1.00 |
| job.languages | 1.00 | 1.00 | 1.00 |

All single values and attributes are 1.00 (is_student, education level / field / status, CEFR
levels, experience months, preferences, job_type, remote_mode, city, country, education
requirement, student requirement, min years), **except job skill modality: 0.83 (19/23)**.
Evidence: 79/79 quotes grounded. About 35 s per document (the model runs 45% on CPU on 4 GB VRAM).

Remaining errors: FastAPI (cv_01) and Firebase (cv_02, project line) missed; "Javi" (Croatian
inflection of "Java") extracted as a skill name; job_04's "Our stack: Python, Airflow, Spark"
marked `required` instead of `unspecified`.

**Caveat:** the prompts were tuned on these same 10 documents (3 rounds: run 1 → run 3 took CV skill
recall from 0.90 to 0.95, city from 0/5 to 5/5, CEFR level from 0.86 to 1.00), so the numbers are
optimistic. The next step is a held-out set from the real datasets.

## Notes for other groups

- **Group 2:** skill names are raw terms as written (`"Vue.js"`, and in Croatian text possibly
  inflected forms like `"Javi"`, `"Excela"`). Normalization and ESCO linking are yours. The prompt
  asks for `occupation` without seniority words, but the 4B model usually copies the title
  (`"Machine Learning Intern"`), so strip intern / junior / senior before occupation mapping.
- **Group 4 (`schemas/models.py` on `group04`):** our output can contain `null` where your models
  currently require a value, as the contract's "unknown is not false" rule requires: language
  `level` (ad says just "German is a plus"), education `level` / `status` (e.g. "degree in CS or
  related"), and `expected_graduation` as `"YYYY"` when the month is not given. Evidence offsets are
  never null. Let's agree on one shared models file on `dev`.
