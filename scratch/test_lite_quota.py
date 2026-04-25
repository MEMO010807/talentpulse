import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

async def test_quota(model_name):
    model = genai.GenerativeModel(model_name)
    success = 0
    print(f"Testing {model_name}...")
    for i in range(25):
        try:
            res = await asyncio.to_thread(
                model.generate_content, ["Hello"]
            )
            success += 1
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"Failed at {i}: {e}")
            break
    print(f"{model_name}: {success}/25 succeeded")

async def run_tests():
    await test_quota("gemini-2.5-flash-lite")
    await test_quota("gemini-flash-lite-latest")

asyncio.run(run_tests())
