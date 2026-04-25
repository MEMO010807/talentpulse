# TalentPulse — Complete Architecture & Implementation Reference

> **Purpose of this document:** This is a full technical dump of everything built so far for the TalentPulse hackathon project. Use it to discuss improvements, catch bugs, or suggest changes with any LLM.

---

## 1. What TalentPulse Does

A recruiter pastes a Job Description → the system automatically:
1. **Parses the JD** → extracts skills, experience, domain via Gemini
2. **Matches 20 candidates** → scores each 0-100 with skill gap analysis
3. **Simulates outreach** → generates 4-turn recruiter↔candidate conversations for top 5
4. **Ranks the shortlist** → final_score = (match × 0.6) + (interest × 0.4)

All in under 60 seconds. Hackathon submission for **Catalyst Hackathon — Deccan AI Experts**.

---

## 2. Folder Structure (current state)

```
talentpulse/
├── main.py                     # FastAPI backend — all pipeline logic (388 lines)
├── candidates.json             # 20 mock candidates (Indian tech market)
├── requirements.txt            # Python deps
├── .env                        # GEMINI_API_KEY (gitignored)
├── .env.example                # Template
├── .gitignore
├── frontend/
│   └── index.html              # Single-file frontend (345 lines, inline CSS+JS)
├── sample_inputs/
│   └── sample_jd.txt           # Sample Sr. Backend Engineer JD (fintech)
├── sample_outputs/
│   └── sample_output.json      # Pre-computed API response (used by demo mode)
└── README.md                   # Setup & usage docs
```

---

## 3. Tech Stack

| Layer      | Tech                                          |
|------------|-----------------------------------------------|
| Backend    | Python 3.10+, FastAPI, Uvicorn                |
| AI Model   | Google Gemini 1.5 Flash (`google-generativeai` SDK) |
| Frontend   | Vanilla HTML/CSS/JS (single file, no build)   |
| Data       | Static JSON file (20 candidates)              |
| Config     | `.env` + `python-dotenv`                      |

**Dependencies** (`requirements.txt`):
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
google-generativeai==0.7.2
python-dotenv==1.0.1
pydantic==2.7.1
httpx==0.27.0
```

---

## 4. Backend Architecture (`main.py`)

### 4.1 Startup & Config
- Loads `.env` for `GEMINI_API_KEY`
- Initializes `genai.GenerativeModel("gemini-1.5-flash")`
- Loads `candidates.json` and `sample_outputs/sample_output.json` into memory
- CORS enabled for all origins (frontend calls API cross-origin)
- Mounts `frontend/` as static files at `/frontend`

### 4.2 Endpoints

| Method | Path             | Purpose                                      |
|--------|------------------|----------------------------------------------|
| GET    | `/health`        | Returns `{"status": "ok"}`                   |
| GET    | `/api/candidates`| Returns all 20 raw candidates                |
| GET    | `/api/demo`      | Returns pre-computed sample output (no API)   |
| POST   | `/api/analyze`   | Full pipeline — accepts `{"job_description": "..."}` |

### 4.3 Pipeline Functions

#### `extract_json(text) → dict`
Robust JSON extractor that handles Gemini responses with:
- Direct JSON parse
- Markdown code fence stripping (` ```json ... ``` `)
- Last-resort brace matching (`{...}` or `[...]`)

#### `gemini_call_with_retry(prompt_parts, max_retries=3) → str`
Wraps `model.generate_content()` with exponential backoff:
- Attempt 1: immediate
- Attempt 2: wait 4s
- Attempt 3: wait 8s
- Only retries on 429 (rate limit) errors

#### `parse_jd(jd_text) → dict` (Step 1)
**Prompt strategy:** System prompt instructs Gemini to return strict JSON with fields:
```json
{
  "role_title": "string",
  "required_skills": ["skill1", ...],
  "preferred_skills": ["skill1", ...],
  "min_experience_years": number,
  "education_requirement": "string",
  "role_type": "full-time | contract | internship",
  "domain": "string",
  "key_responsibilities": ["string", ...]
}
```
**Fallback:** If API fails, returns a generic "Software Engineer" object with empty arrays.

#### `_score_single_candidate(parsed_jd, candidate) → dict` (Step 2 helper)
**Prompt strategy:** Sends full parsed JD + full candidate profile as JSON. Asks for:
```json
{
  "match_score": 0-100,
  "matched_skills": [...],
  "missing_skills": [...],
  "explanation": "2-3 sentences"
}
```

#### `match_candidates(parsed_jd) → list[dict]` (Step 2)
- Runs **sequentially** (one candidate at a time) with 1s delay between calls
- This is critical for free-tier rate limits (was parallel batches of 5, caused 429s)
- Sorts by match_score descending, returns top 5

#### `_simulate_single_outreach(parsed_jd, match_result) → dict` (Step 3 helper)
**Prompt strategy:** Single Gemini call per candidate that generates both conversation AND interest score:
```json
{
  "conversation": [
    {"role": "recruiter", "message": "..."},
    {"role": "candidate", "message": "..."},
    {"role": "recruiter", "message": "..."},
    {"role": "candidate", "message": "..."}
  ],
  "interest_score": 0-100,
  "interest_label": "Hot | Warm | Cold",
  "interest_reasoning": "1-2 sentences"
}
```
Includes candidate name, role, experience, skills, availability, summary in the user prompt for personalized conversations.

#### `simulate_outreach(parsed_jd, top_matches) → list[dict]` (Step 3)
- Sequential with 1s delays (same rate-limit strategy as matching)

#### `rank_candidates(enriched) → list[dict]` (Step 4)
- `final_score = (match_score * 0.6) + (interest_score * 0.4)`
- Sorts descending, assigns rank 1-5
- Returns enriched objects with all fields

### 4.4 Total Gemini API Calls Per Request
- 1 (JD parse) + 20 (candidate matching) + 5 (outreach) = **26 API calls**
- With 1s delays: ~25s minimum execution time
- Free tier: 15 RPM → sequential calls stay under limit

---

## 5. Candidate Database (`candidates.json`)

20 candidates, each with:
```json
{
  "id": "c001", "name": "...", "current_role": "...", "current_company": "...",
  "years_experience": 5, "education": "...",
  "skills": ["Python", "FastAPI", ...],
  "domain": "backend|frontend|fullstack|ml|devops|data",
  "location": "...", "availability": "immediate|30 days|60 days|not looking",
  "expected_ctc_lpa": 18, "linkedin_url": "...", "summary": "..."
}
```

**Distribution by design:**
- Strong backend matches: c001 (Arjun/Razorpay), c003 (Rahul/CRED), c008 (Meera/PhonePe), c014 (Tanvi/Razorpay), c013 (Siddharth/Zerodha), c016 (Pooja/Slice), c019 (Manish/Paytm)
- Medium matches: c002 (Priya/Flipkart), c017 (Harsh/Amazon), c005 (Vikram/ML)
- Weak/domain mismatch: c009 (Aditya/Frontend), c012 (Nisha/Junior), c020 (Lakshmi/Analyst), c006 (Ananya/Java)
- Not looking: c009, c011, c018

---

## 6. Frontend (`frontend/index.html`)

### Design System
- **Theme:** Dark bg `#0a0a0f`, blue `#2563eb`, cyan `#06b6d4`
- **Fonts:** DM Sans (body), Space Grotesk (headings/scores) via Google Fonts CDN
- **Components:** Textarea, buttons, loader spinner, parsed JD card, candidate cards, chat UI

### UI Flow
1. Input section: textarea + "Scout Candidates →" + "⚡ Try Demo"
2. Loading: spinning circle + cycling step messages every 2.5s
3. Parsed JD: collapsible card showing extracted fields + skill tags
4. Ranked Shortlist: 5 candidate cards with:
   - Rank badge (gold #1, silver #2, bronze #3)
   - Name, role, company, location
   - Animated score bars (match=blue, interest=cyan) with 800ms CSS transition
   - Matched skills (green tags), missing skills (red tags)
   - Match explanation (italic)
   - Expandable conversation (chat bubble UI, recruiter left, candidate right)
   - Interest analysis callout

### JS Functions
- `handleScout()` → POST `/api/analyze` with JD text
- `handleDemo()` → GET `/api/demo` (pre-computed, no Gemini needed)
- `renderResults(data)` → builds all DOM elements
- `toggleConv(id)` → expand/collapse conversation via CSS max-height
- `esc(s)` → XSS-safe text escaping

---

## 7. Known Issues & Problems Encountered

### Rate Limiting (CRITICAL)
- Free-tier Gemini has ~15 RPM limit
- Original design used `asyncio.gather` with batches of 5 → immediately hit 429
- **Fixed:** All calls now sequential with 1s delays + retry with exponential backoff
- **Daily quota exhaustion:** First parallel run burned through the daily quota on `gemini-2.0-flash`
- **Mitigation:** Switched to `gemini-1.5-flash` (separate quota bucket) + added demo mode

### Windows Encoding
- `print("⚠️")` crashes on Windows cp1252 console
- **Fixed:** Replaced emoji with ASCII `[WARNING]`

### Demo Mode (Safety Net)
- `/api/demo` endpoint serves `sample_outputs/sample_output.json` directly
- "Try Demo" button on frontend calls this — zero API dependency
- Guarantees a working demo even if Gemini is down or quota exhausted

---

## 8. API Response Shape

```json
{
  "parsed_jd": {
    "role_title": "Senior Backend Engineer",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "AWS"],
    "preferred_skills": ["Kafka", "Kubernetes", "Fintech experience"],
    "min_experience_years": 4,
    "education_requirement": "B.Tech/B.E. or equivalent",
    "role_type": "full-time",
    "domain": "backend",
    "key_responsibilities": ["Design RESTful APIs", "Own payment pipelines", ...]
  },
  "shortlist": [
    {
      "rank": 1,
      "candidate": { /* full candidate object */ },
      "match_score": 95,
      "interest_score": 78,
      "final_score": 88.2,
      "matched_skills": ["Python", "FastAPI", "PostgreSQL", ...],
      "missing_skills": [],
      "match_explanation": "Arjun is an excellent match with all required...",
      "conversation": [
        {"role": "recruiter", "message": "Hi Arjun! ..."},
        {"role": "candidate", "message": "Hey, thanks for reaching out! ..."},
        {"role": "recruiter", "message": "Great! The role involves..."},
        {"role": "candidate", "message": "That sounds compelling..."}
      ],
      "interest_label": "Hot",
      "interest_reasoning": "Shows strong interest due to domain alignment..."
    }
    // ... 4 more candidates
  ],
  "pipeline_summary": "Analyzed JD for 'Senior Backend Engineer'. Screened 20 candidates..."
}
```

---

## 9. Gemini Prompts Used (exact text)

### JD Parser System Prompt
```
You are a JD parser. Extract structured information from job descriptions.
Return ONLY valid JSON with these exact fields:
{
  "role_title": "string",
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", ...],
  "min_experience_years": number,
  "education_requirement": "string",
  "role_type": "full-time | contract | internship",
  "domain": "string (e.g. backend, ML, frontend, data, devops)",
  "key_responsibilities": ["string", ...]
}
Return only the JSON object, no markdown, no explanation.
```

### Candidate Matcher System Prompt
```
You are a talent matching engine. Given a parsed job description and a candidate profile, evaluate how well the candidate matches the role.
Return ONLY valid JSON:
{
  "match_score": number (0-100),
  "matched_skills": ["skill1", ...],
  "missing_skills": ["skill1", ...],
  "explanation": "2-3 sentence human-readable explanation of why this score was given"
}
Return only the JSON object, no markdown, no explanation.
```

### Outreach Simulator System Prompt
```
You are simulating a realistic recruiter-candidate outreach conversation AND scoring the candidate's interest level.

Recruiter persona: Professional, friendly, brief messages.
Candidate persona: Based on the candidate profile provided. Respond authentically — some candidates are enthusiastic, some are passive, some are not interested. Make the Interest Score feel earned, not random.

Generate a 4-turn conversation (recruiter, candidate, recruiter, candidate) and then score the candidate's interest.

Return ONLY valid JSON in this exact format:
{
  "conversation": [
    { "role": "recruiter", "message": "string" },
    { "role": "candidate", "message": "string" },
    { "role": "recruiter", "message": "string" },
    { "role": "candidate", "message": "string" }
  ],
  "interest_score": number (0-100),
  "interest_label": "Hot | Warm | Cold",
  "interest_reasoning": "1-2 sentence explanation of why this interest level was assigned"
}
Return only the JSON object, no markdown, no explanation.
```

---

## 10. Scoring Formula

```
final_score = (match_score × 0.6) + (interest_score × 0.4)
```

- **Match Score (60%)**: Skill/experience alignment with JD — the primary hiring signal
- **Interest Score (40%)**: Simulated candidate engagement likelihood — ensures shortlist is actionable
- **Interest Labels**: Hot (70-100), Warm (40-69), Cold (0-39)

---

## 11. How to Run

```bash
cd talentpulse
pip install -r requirements.txt
# Edit .env with your GEMINI_API_KEY
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/frontend/index.html
```

---

## 12. Discussion Points for Review

Things to consider improving:

1. **Rate limit strategy**: Currently 26 sequential API calls with 1s delays (~25s). Could batch candidate data into fewer, larger prompts (e.g., score 5 candidates per call) to reduce total calls from 26 to ~6.

2. **Prompt engineering**: Are the prompts optimal? Should we use structured output / JSON mode instead of hoping Gemini returns valid JSON?

3. **Scoring consistency**: Gemini may score differently each run. Should we add temperature=0 for deterministic results? Should we calibrate scores with few-shot examples in the prompt?

4. **Frontend polish**: The UI works but could benefit from more animations, a progress bar showing actual pipeline stage, or a skeleton loading state.

5. **Error handling**: Currently silent fallbacks on API errors. Should the frontend show partial results or explicitly indicate which candidates failed?

6. **Candidate database**: Static 20 candidates. For the demo, is this sufficient or should we add more variety?

7. **Model choice**: Currently `gemini-1.5-flash` (switched from 2.0-flash due to quota issues). Should we use `gemini-2.0-flash-lite` or stick with 1.5-flash?

8. **Caching**: No caching currently. Same JD analyzed twice = 26 more API calls. Worth adding?

9. **The 60/40 weighting**: Is this the right split? Should it be configurable from the frontend?

10. **Security**: API key is in `.env`, CORS is `*`. Fine for hackathon, but worth noting.
