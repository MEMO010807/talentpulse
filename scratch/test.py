import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

async def test():
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            ["Hello world"]
        )
        print("Response:", response.text)
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(test())
