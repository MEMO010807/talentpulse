# TalentPulse — Complete Architecture & Implementation Reference

> **Purpose of this document:** This is a full technical dump of everything built so far for the TalentPulse hackathon project. Use it to discuss improvements, catch bugs, or suggest changes with any LLM.

---

## 1. What TalentPulse Does

A recruiter pastes a Job Description → the system automatically:
1. **Parses the JD** → extracts skills, experience, domain via Gemini
2. **Matches 20 candidates** → scores each 0-100 with skill gap analysis
3. **Simulates outreach** → generates 4-turn recruiter↔candidate conversations for top 5
4. **Ranks the shortlist** → final_score = (match × 0.6) + (interest × 0.4)

All in under 15 seconds. Hackathon submission for **Catalyst Hackathon — Deccan AI Experts**.

---

## 2. Folder Structure (current state)

```
talentpulse/
├── main.py                     # FastAPI backend — all pipeline logic
├── candidates.json             # 20 mock candidates (Indian tech market)
├── requirements.txt            # Python deps
├── .env                        # GEMINI_API_KEY (gitignored)
├── .env.example                # Template
├── .gitignore
├── frontend/
│   └── index.html              # Single-file frontend (inline CSS+JS)
├── sample_inputs/
│   └── sample_jd.txt           # Sample Sr. Backend Engineer JD (fintech)
├── sample_outputs/
│   └── sample_output.json      # Pre-computed API response (used by demo mode)
├── CLAUDE_DEBUG_REPORT.md      # Deprecated architecture quota report
└── README.md                   # Setup & usage docs
```

---

## 3. Tech Stack

| Layer      | Tech                                          |
|------------|-----------------------------------------------|
| Backend    | Python 3.10+, FastAPI, Uvicorn                |
| AI Model   | Google Gemini Flash Lite (`google-generativeai` SDK) |
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
- Initializes `genai.GenerativeModel("gemini-flash-lite-latest")`
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

### 4.3 Pipeline Functions (3-Call Architecture)

To bypass the strict `15 RPM` limits on the free tier, the system leverages Gemini's 1M context window and `response_mime_type="application/json"` to batch operations. The entire pipeline executes in exactly **3 API calls**.

#### `gemini_json_call(prompt, max_retries=4) → dict | list`
Wraps `model.generate_content()` with `GenerationConfig(response_mime_type="application/json")`.
- Directly returns parsed JSON.
- Implements exponential backoff on `429 Too Many Requests`.

#### `parse_jd(jd_text) → dict` (Call 1)
Extracts structured JSON fields from the JD: `role_title`, `required_skills`, `preferred_skills`, `min_experience_years`, `education_requirement`, `role_type`, `domain`, `key_responsibilities`.

#### `match_candidates_batch(parsed_jd) → list[dict]` (Call 2)
- Replaces the old 20-call loop. 
- Passes the sanitized JD and an array of all 20 stripped candidate objects to Gemini.
- Returns a single JSON array of 20 score objects (`match_score`, `matched_skills`, `missing_skills`, `explanation`).
- Calculates deterministic metrics (`skills_coverage_pct`, `experience_gap_years`, `domain_match`) locally in Python to save tokens.

#### `simulate_outreach_batch(parsed_jd, top_matches) → list[dict]` (Call 3)
- Replaces the old 5-call loop.
- Simulates the 5 conversations simultaneously by passing an array of the top 5 candidates.
- Returns an array of 5 conversation objects (`conversation`, `interest_score`, `interest_label`, `interest_reasoning`).
- Injects a hard constraint directly into the input data if a candidate's availability is "not looking", forcing the `interest_score` to 0-20.

#### `rank_candidates(enriched) → list[dict]` (Local execution)
- `final_score = (match_score * 0.6) + (interest_score * 0.4)`
- Sorts descending, assigns rank 1-5
- Returns enriched objects with all fields

### 4.4 Total Gemini API Calls Per Request
- 1 (JD parse) + 1 (all 20 candidates matching) + 1 (all 5 outreach) = **3 API calls**
- Time: ~8-15 seconds execution time
- Fits perfectly within the 15 RPM free-tier quota.

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

---

## 6. Frontend (`frontend/index.html`)

### Design System
- **Theme:** Dark bg `#0a0a0f`, blue `#2563eb`, cyan `#06b6d4`
- **Fonts:** DM Sans (body), Space Grotesk (headings/scores) via Google Fonts CDN

### UI Flow
1. Input section: textarea + "Scout Candidates →" + "⚡ Try Demo"
2. Loading: spinning circle + cycling step messages every 2.5s
3. Parsed JD: collapsible card showing extracted fields + skill tags
4. Ranked Shortlist: 5 candidate cards with animated score bars, skill tags, expandable chat UI, and interest analysis callout.

---

## 7. Known Issues & Problems Encountered

### Rate Limiting (Resolved)
- Free-tier Gemini Flash Lite has a `15 RPM` limit (and Gemini 2.5 Flash has a `20 RPD` limit).
- Original design used 26 sequential calls with delays → exhausted quotas instantly.
- **Fixed:** Consolidated the 26 calls into 3 batch calls using JSON arrays.

### JSON Parsing Fragility (Resolved)
- Original design used regex to strip markdown fences from standard Gemini text output.
- **Fixed:** Switched to `response_mime_type="application/json"` which completely guarantees structured output.

### Windows Encoding (Resolved)
- `print("⚠️")` crashes on Windows cp1252 console
- **Fixed:** Replaced emoji with ASCII `[WARNING]`

---

## 8. API Response Shape

```json
{
  "parsed_jd": { ... },
  "shortlist": [
    {
      "rank": 1,
      "candidate": { /* full candidate object */ },
      "match_score": 95,
      "interest_score": 78,
      "final_score": 88.2,
      "matched_skills": ["Python", "FastAPI", "PostgreSQL", ...],
      "missing_skills": [],
      "match_explanation": "...",
      "conversation": [
        {"role": "recruiter", "message": "Hi Arjun! ..."},
        {"role": "candidate", "message": "Hey, thanks for reaching out! ..."}
      ],
      "interest_label": "Hot",
      "interest_reasoning": "..."
    }
    // ... 4 more candidates
  ],
  "pipeline_summary": "..."
}
```

---

## 9. Scoring Formula

```
final_score = (match_score × 0.6) + (interest_score × 0.4)
```

- **Match Score (60%)**: Skill/experience alignment with JD — the primary hiring signal
- **Interest Score (40%)**: Simulated candidate engagement likelihood — ensures shortlist is actionable
- **Interest Labels**: Hot (70-100), Warm (40-69), Cold (0-39)

---

## 10. How to Run

```bash
cd talentpulse
pip install -r requirements.txt
# Edit .env with your GEMINI_API_KEY
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/frontend/index.html
```
