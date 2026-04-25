"""
TalentPulse — AI-Powered Talent Scouting & Engagement Agent
FastAPI backend with Gemini 2.0 Flash integration.
"""

import os
import json
import asyncio
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print("[WARNING] GEMINI_API_KEY not set. Add it to .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------------------------------------------------------
# Load candidate database
# ---------------------------------------------------------------------------
CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "candidates.json")
with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    CANDIDATES: list[dict[str, Any]] = json.load(f)

SAMPLE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_outputs", "sample_output.json")
SAMPLE_OUTPUT: dict[str, Any] = {}
if os.path.isfile(SAMPLE_OUTPUT_PATH):
    with open(SAMPLE_OUTPUT_PATH, "r", encoding="utf-8") as f:
        SAMPLE_OUTPUT = json.load(f)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TalentPulse API",
    description="AI-Powered Talent Scouting & Engagement Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    job_description: str

# ---------------------------------------------------------------------------
# Utility — robust JSON extraction from Gemini responses
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    """Extract JSON from a Gemini response that might contain markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last-resort: find first { … } or [ … ]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from Gemini response:\n{text[:500]}")


async def gemini_call_with_retry(prompt_parts: list, max_retries: int = 3) -> str:
    """Call Gemini with exponential backoff retry for rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt_parts,
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries - 1:
                wait = (2 ** attempt) * 4  # 4s, 8s, 16s
                print(f"Rate limited (attempt {attempt+1}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

# ---------------------------------------------------------------------------
# Pipeline Step 1 — Parse JD
# ---------------------------------------------------------------------------
async def parse_jd(jd_text: str) -> dict:
    """Call Gemini to extract structured fields from a raw JD."""
    system_prompt = (
        "You are a JD parser. Extract structured information from job descriptions.\n"
        "Return ONLY valid JSON with these exact fields:\n"
        "{\n"
        '  "role_title": "string",\n'
        '  "required_skills": ["skill1", "skill2", ...],\n'
        '  "preferred_skills": ["skill1", ...],\n'
        '  "min_experience_years": number,\n'
        '  "education_requirement": "string",\n'
        '  "role_type": "full-time | contract | internship",\n'
        '  "domain": "string (e.g. backend, ML, frontend, data, devops)",\n'
        '  "key_responsibilities": ["string", ...]\n'
        "}\n"
        "Return only the JSON object, no markdown, no explanation."
    )

    try:
        text = await gemini_call_with_retry(
            [system_prompt, f"Job Description:\n{jd_text}"],
        )
        return extract_json(text)
    except Exception as e:
        print(f"JD parsing error: {e}")
        return {
            "role_title": "Software Engineer",
            "required_skills": [],
            "preferred_skills": [],
            "min_experience_years": 0,
            "education_requirement": "Not specified",
            "role_type": "full-time",
            "domain": "backend",
            "key_responsibilities": [],
        }

# ---------------------------------------------------------------------------
# Pipeline Step 2 — Match Candidates
# ---------------------------------------------------------------------------
async def _score_single_candidate(parsed_jd: dict, candidate: dict) -> dict:
    """Score a single candidate against the parsed JD via Gemini."""
    system_prompt = (
        "You are a talent matching engine. Given a parsed job description and a candidate profile, "
        "evaluate how well the candidate matches the role.\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "match_score": number (0-100),\n'
        '  "matched_skills": ["skill1", ...],\n'
        '  "missing_skills": ["skill1", ...],\n'
        '  "explanation": "2-3 sentence human-readable explanation of why this score was given"\n'
        "}\n"
        "Return only the JSON object, no markdown, no explanation."
    )

    user_prompt = (
        f"Parsed Job Description:\n{json.dumps(parsed_jd, indent=2)}\n\n"
        f"Candidate Profile:\n{json.dumps(candidate, indent=2)}"
    )

    try:
        text = await gemini_call_with_retry(
            [system_prompt, user_prompt],
        )
        result = extract_json(text)
        result["candidate"] = candidate
        return result
    except Exception as e:
        print(f"Matching error for {candidate.get('name', '?')}: {e}")
        return {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": parsed_jd.get("required_skills", []),
            "explanation": "Scoring failed due to an API error.",
            "candidate": candidate,
        }


async def match_candidates(parsed_jd: dict) -> list[dict]:
    """Score all candidates sequentially to respect free-tier rate limits."""
    all_results: list[dict] = []

    for idx, candidate in enumerate(CANDIDATES):
        result = await _score_single_candidate(parsed_jd, candidate)
        all_results.append(result)
        # Small delay between each call to stay under rate limits
        if idx < len(CANDIDATES) - 1:
            await asyncio.sleep(1.0)

    # Sort descending by match_score and take top 5
    all_results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return all_results[:5]

# ---------------------------------------------------------------------------
# Pipeline Step 3 — Conversational Outreach Simulation
# ---------------------------------------------------------------------------
async def _simulate_single_outreach(parsed_jd: dict, match_result: dict) -> dict:
    """Simulate recruiter↔candidate conversation for one candidate."""
    candidate = match_result["candidate"]

    system_prompt = (
        "You are simulating a realistic recruiter-candidate outreach conversation AND scoring the candidate's interest level.\n\n"
        "Recruiter persona: Professional, friendly, brief messages.\n"
        f"Candidate persona: Based on the candidate profile provided. Respond authentically — "
        "some candidates are enthusiastic, some are passive, some are not interested. "
        "Make the Interest Score feel earned, not random.\n\n"
        "Generate a 4-turn conversation (recruiter, candidate, recruiter, candidate) and then score the candidate's interest.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "conversation": [\n'
        '    { "role": "recruiter", "message": "string" },\n'
        '    { "role": "candidate", "message": "string" },\n'
        '    { "role": "recruiter", "message": "string" },\n'
        '    { "role": "candidate", "message": "string" }\n'
        "  ],\n"
        '  "interest_score": number (0-100),\n'
        '  "interest_label": "Hot | Warm | Cold",\n'
        '  "interest_reasoning": "1-2 sentence explanation of why this interest level was assigned"\n'
        "}\n"
        "Return only the JSON object, no markdown, no explanation."
    )

    user_prompt = (
        f"Job Role: {parsed_jd.get('role_title', 'Software Engineer')}\n"
        f"Company Domain: Fintech startup in Bengaluru\n"
        f"Required Skills: {', '.join(parsed_jd.get('required_skills', []))}\n\n"
        f"Candidate Name: {candidate.get('name')}\n"
        f"Current Role: {candidate.get('current_role')} at {candidate.get('current_company')}\n"
        f"Years of Experience: {candidate.get('years_experience')}\n"
        f"Key Skills: {', '.join(candidate.get('skills', []))}\n"
        f"Location: {candidate.get('location')}\n"
        f"Availability: {candidate.get('availability')}\n"
        f"Summary: {candidate.get('summary')}\n"
    )

    try:
        text = await gemini_call_with_retry(
            [system_prompt, user_prompt],
        )
        result = extract_json(text)
        match_result.update(result)
        return match_result
    except Exception as e:
        print(f"Outreach simulation error for {candidate.get('name', '?')}: {e}")
        match_result.update(
            {
                "conversation": [],
                "interest_score": 50,
                "interest_label": "Warm",
                "interest_reasoning": "Simulation unavailable due to an API error.",
            }
        )
        return match_result


async def simulate_outreach(parsed_jd: dict, top_matches: list[dict]) -> list[dict]:
    """Run outreach simulation sequentially to respect free-tier rate limits."""
    results = []
    for idx, m in enumerate(top_matches):
        result = await _simulate_single_outreach(parsed_jd, m)
        results.append(result)
        if idx < len(top_matches) - 1:
            await asyncio.sleep(1.0)
    return results

# ---------------------------------------------------------------------------
# Pipeline Step 4 — Final Ranking
# ---------------------------------------------------------------------------
def rank_candidates(enriched: list[dict]) -> list[dict]:
    """Combine match + interest scores and produce final ranked list."""
    for item in enriched:
        ms = item.get("match_score", 0)
        is_ = item.get("interest_score", 0)
        item["final_score"] = round(ms * 0.6 + is_ * 0.4, 1)

    enriched.sort(key=lambda x: x["final_score"], reverse=True)

    ranked = []
    for idx, item in enumerate(enriched, 1):
        ranked.append(
            {
                "rank": idx,
                "candidate": item["candidate"],
                "match_score": item.get("match_score", 0),
                "interest_score": item.get("interest_score", 0),
                "final_score": item["final_score"],
                "matched_skills": item.get("matched_skills", []),
                "missing_skills": item.get("missing_skills", []),
                "match_explanation": item.get("explanation", ""),
                "conversation": item.get("conversation", []),
                "interest_label": item.get("interest_label", "Warm"),
                "interest_reasoning": item.get("interest_reasoning", ""),
            }
        )
    return ranked

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/candidates")
async def get_candidates():
    return CANDIDATES


@app.get("/api/demo")
async def demo():
    """Return pre-computed sample output — works without Gemini API."""
    if not SAMPLE_OUTPUT:
        raise HTTPException(status_code=404, detail="Sample output not found.")
    return SAMPLE_OUTPUT


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Add it to the .env file.",
        )

    # Step 1 — Parse JD
    parsed_jd = await parse_jd(req.job_description)

    # Step 2 — Match candidates
    top_matches = await match_candidates(parsed_jd)

    # Step 3 — Simulate outreach
    enriched = await simulate_outreach(parsed_jd, top_matches)

    # Step 4 — Rank
    shortlist = rank_candidates(enriched)

    pipeline_summary = (
        f"Analyzed JD for '{parsed_jd.get('role_title', 'Unknown Role')}'. "
        f"Screened {len(CANDIDATES)} candidates, shortlisted top 5. "
        f"Simulated recruiter outreach for each shortlisted candidate. "
        f"Final ranking uses 60% match score + 40% interest score."
    )

    return {
        "parsed_jd": parsed_jd,
        "shortlist": shortlist,
        "pipeline_summary": pipeline_summary,
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
