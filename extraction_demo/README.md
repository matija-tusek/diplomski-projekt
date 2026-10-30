# Group 1: extraction demo (Ollama + Qwen3-4B-Instruct)

Turns a raw CV or job ad into structured JSON with a local LLM. The output is a subset of
`CandidateProfile` / `JobProfile` from the shared schema v1.0.

Detaljne upute i objašnjenje na hrvatskom: [UPUTE.md](UPUTE.md).

How it works:
1. Pydantic models define the output shape; `Model.model_json_schema()` is passed to Ollama as
   `format`, so the model can only produce JSON matching the schema (enums included).
2. The LLM quotes an evidence sentence for every item.
3. `extract.py` finds each quote in the source text and adds `char_start` / `char_end`.
   If a quote is not found verbatim, offsets are `null`, which flags a paraphrase or hallucination.

## Setup (Windows)

```powershell
winget install --id Ollama.Ollama -e     # or the installer from https://ollama.com/download
ollama pull qwen3:4b-instruct            # ~2.5 GB
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe extract.py cv  samples/cv.txt
.\.venv\Scripts\python.exe extract.py job samples/job.txt
```

The JSON is printed and saved to `output/<name>.json`, followed by a line like
`Evidence found verbatim in source: 12/13`.
