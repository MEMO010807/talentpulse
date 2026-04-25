import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")
deterministic_config = genai.types.GenerationConfig(temperature=0.0)

async def test():
    system_prompt = "You are a JD parser. Return JSON."
    user_prompt = "Hello"
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            [system_prompt, user_prompt],
            generation_config=deterministic_config
        )
        print("Success:", response.text)
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(test())
