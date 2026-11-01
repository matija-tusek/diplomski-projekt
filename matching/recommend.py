from pathlib import Path

import ollama
import schemas.models as models
from pydantic import TypeAdapter

LLM_MODEL = "qwen3:4b-instruct"
DATA_PATH = Path("matching/mock_data")

jobs = TypeAdapter(list[models.NormalizedProfile]).validate_json((DATA_PATH / "jobs.json").read_bytes())

for job in jobs:
    response = ollama.chat(model=LLM_MODEL, messages=[
        {
            "role": "system",
            "content": "You will receive a job posting as JSON. Write a 2-3 sentence description in plain prose. "
        },
        {
            "role": "user",
            "content": job.model_dump_json()
        }
    ])
    print(response.message.content)