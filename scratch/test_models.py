import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

async def test_model(model_name):
    try:
        model = genai.GenerativeModel(model_name)
        response = await asyncio.to_thread(
            model.generate_content,
            ["Hello"]
        )
        print(f"SUCCESS: {model_name}")
    except Exception as e:
        print(f"FAILED: {model_name} -> {repr(e)}")

async def run_all():
    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-flash-latest",
        "gemini-2.5-flash"
    ]
    for m in models:
        await test_model(m)

asyncio.run(run_all())
