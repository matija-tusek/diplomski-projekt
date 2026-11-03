import schemas.models as models
import ollama

instructions_message = """
You will receive a MatchResponse object as JSON.
Write a 2-3 sentence description in plain prose. Describe which are key skills that candidate has and which crutial skills are missing.
You must mention actual skills present in the object and give concrete exaples.
"""

def describe_match(match_result : models.MatchResult, llm_model : str, language : str):
    response = ollama.chat(model=llm_model, messages=[
        {
            "role": "system",
            "content": instructions_message + f" Write only in language with this ISO country code: {language}."
        },
        {
            "role": "user",
            "content": match_result.model_dump_json()
        }
    ])
    return response.message.content