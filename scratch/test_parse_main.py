import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

import main

async def test():
    jd = "We are looking for a Senior Backend Engineer with 4+ years of experience in Python, FastAPI, PostgreSQL, and AWS."
    parsed = await main.parse_jd(jd)
    print("PARSED:", parsed)

asyncio.run(test())
