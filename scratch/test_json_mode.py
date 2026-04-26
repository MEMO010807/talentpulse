import os, json, asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-flash-lite-latest")

config = GenerationConfig(temperature=0.0, response_mime_type="application/json")

# Small test: score 3 candidates
prompt = """Score these 3 candidates against the JD. Return a JSON array of 3 objects.
Each object: {"candidate_id": "string", "match_score": integer 0-100, "explanation": "string"}

JD: Senior Backend Engineer, requires Python, FastAPI, PostgreSQL, 4+ years.

Candidates:
[{"id": "c1", "name": "Alice", "skills": ["Python", "FastAPI", "PostgreSQL"], "years_experience": 5},
 {"id": "c2", "name": "Bob", "skills": ["Java", "Spring"], "years_experience": 2},
 {"id": "c3", "name": "Carol", "skills": ["Python", "Django"], "years_experience": 6}]

Return JSON array now."""

async def test():
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt, generation_config=config),
            timeout=30.0
        )
        print("RAW TEXT:")
        print(repr(raw.text[:2000]))
        parsed = json.loads(raw.text)
        print("\nPARSED OK:", json.dumps(parsed, indent=2)[:500])
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(test())
