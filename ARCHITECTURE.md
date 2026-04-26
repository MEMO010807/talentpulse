# TalentPulse — Complete Architecture & Implementation Reference

> **Purpose:** Full technical reference for the TalentPulse hackathon project. Covers every architectural decision, function, prompt, and scoring formula.

---

## 1. What TalentPulse Does

A recruiter pastes a Job Description → the system automatically:
1. **Parses the JD** → extracts skills, experience, domain via Gemini (Call 1)
2. **Batch-scores 20 candidates** → all in one prompt, returns match scores 0–100 (Call 2)
3. **Batch-simulates outreach** → 4-turn conversations for top 5, with interest scores (Call 3)
4. **Ranks the shortlist** → `final_score = (match × 0.6) + (interest × 0.4)`

**3 API calls total. ~15 seconds end-to-end.** Hackathon submission for **Catalyst Hackathon — Deccan AI Experts**.

---

## 2. Folder Structure

```
talentpulse/
├── main.py                     # FastAPI backend — all pipeline logic
├── candidates.json             # 20 mock candidates (Indian tech market)
├── requirements.txt            # Python deps (no new packages)
├── .env                        # GEMINI_API_KEY (gitignored)
├── .env.example                # Template
├── .gitignore
├── frontend/
│   └── index.html              # Single-file frontend (inline CSS+JS)
├── sample_inputs/
│   └── sample_jd.txt           # Sample Sr. Backend Engineer JD (fintech)
├── sample_outputs/
│   └── sample_output.json      # Pre-computed API response (demo mode)
├── ARCHITECTURE.md             # This file
├── CLAUDE_DEBUG_REPORT.md      # Historical debugging notes
└── README.md                   # Setup & usage docs
```

---

## 3. Tech Stack

| Layer      | Tech                                                    |
|------------|---------------------------------------------------------|
| Backend    | Python 3.10+, FastAPI 0.111.0, Uvicorn 0.29.0          |
| AI Model   | `gemini-flash-lite-latest` via `google-generativeai==0.7.2` |
| Frontend   | Vanilla HTML/CSS/JS (single file, no build step)        |
| Data       | Static JSON file (20 candidates)                        |
| Config     | `.env` + `python-dotenv==1.0.1`                         |

---

## 4. Backend Architecture (`main.py`)

### 4.1 Startup & Config
- Loads `.env` with `load_dotenv(override=True)`
- **Startup guard:** Refuses to start if `GEMINI_API_KEY` is missing (`RuntimeError`)
- Initializes `genai.GenerativeModel("gemini-flash-lite-latest")`
- Loads `candidates.json` and `sample_outputs/sample_output.json` into memory
- CORS enabled for all origins
- Mounts `frontend/` as static files at `/frontend`

### 4.2 Endpoints

| Method | Path             | Purpose                                      |
|--------|------------------|----------------------------------------------|
| GET    | `/health`        | Returns `{"status": "ok"}`                   |
| GET    | `/api/candidates`| Returns all 20 raw candidates                |
| GET    | `/api/demo`      | Returns pre-computed sample output (no API)   |
| POST   | `/api/analyze`   | Full 3-call pipeline                          |

### 4.3 Concurrency & Rate-Limit Protection
- **`asyncio.Semaphore(2)`** guards all Gemini calls — prevents concurrent user requests from exceeding the 15 RPM free-tier quota
- No `asyncio.gather` anywhere — the 3 pipeline calls are inherently sequential
- No `time.sleep` — only `await asyncio.sleep` for non-blocking waits
- No artificial delays between the 3 main calls (they fit within 15 RPM)

### 4.4 Gemini Call Wrapper — Dual-Mode Strategy

`gemini_json_call()` is the single entry point for all Gemini interactions.

**Why dual-mode?** `gemini-flash-lite-latest` intermittently generates malformed JSON when using the SDK's `response_mime_type="application/json"`. The fix:

1. **Attempt 1:** JSON mode (`response_mime_type="application/json"`, `temperature=0.0`)
2. **Attempt 2+:** Falls back to text mode with robust `_extract_json()` regex extraction
3. Each attempt has a **45-second timeout** via `asyncio.wait_for()`
4. Rate-limit errors (`429`) trigger exponential backoff: `8s → 16s → 32s`

**`_extract_json(text)`** handles:
- Direct `json.loads()` parse
- Markdown fence stripping (` ```json ... ``` `)
- Outermost `[...]` or `{...}` brace matching
- Truncated array repair (find last complete `}` and close the array)

### 4.5 Generation Configs

| Config            | Temperature | response_mime_type   | max_output_tokens | Used for           |
|-------------------|-------------|----------------------|-------------------|--------------------|
| `JSON_CONFIG`     | 0.0         | `application/json`   | 8192              | JD parse, scoring  |
| `CREATIVE_CONFIG` | 0.4         | `application/json`   | 8192              | Outreach           |
| `TEXT_CONFIG`      | 0.0         | *(none — text mode)* | 8192              | Scoring fallback   |
| `CREATIVE_TEXT_CONFIG` | 0.4    | *(none — text mode)* | 8192              | Outreach fallback  |

### 4.6 Pipeline Functions (3-Call Architecture)

#### Call 1: `parse_jd(jd_text) → dict`
- Sanitises JD via `sanitise_jd()` (prompt injection defence, code block removal)
- Wraps JD in `<jd>` delimiters to contain injection
- Returns: `role_title`, `required_skills`, `preferred_skills`, `min_experience_years`, `education_requirement`, `role_type`, `domain`, `key_responsibilities`
- **Fallback:** Returns generic "Software Engineer" placeholder on failure

#### Call 2: `match_candidates_batch(parsed_jd) → (top5_list, failed_count)`
- Strips PII fields before sending (no `linkedin_url`, `location`, `expected_ctc_lpa`)
- Sends all 20 candidates + JD summary in one prompt
- Requires `candidate_id` echoed in each output object for ordering validation
- Includes calibration anchors in the prompt (90+, 55-70, below 25)
- **Deterministic blend scoring:**
  - `llm_score`: The model's holistic 0-100 judgment
  - `det_score`: `compute_deterministic_match()` — transparent Python formula:
    - `skills_coverage × 0.60 − exp_penalty + domain_bonus × 0.15`
  - `match_score = round(llm_score × 0.5 + det_score × 0.5)`
- Computes `score_breakdown` locally (skills_coverage_pct, experience_gap_years, domain_match)
- Sorts by `match_score` descending, returns top 5
- **Partial results tolerated:** If model returns <20 items, uses what it got

#### Call 3: `simulate_outreach_batch(parsed_jd, top_matches) → list`
- Sends top 5 candidates with `_index` for ordering
- Injects `_constraint` for "not looking" candidates (forces Cold, 0-20)
- Uses `creative=True` (temperature=0.4) for realistic conversation variation
- **Post-parse enforcement:** Even if LLM ignores the constraint, Python clamps "not looking" to ≤20 and "Cold"
- Each candidate gets: `conversation` (4 turns), `interest_score`, `interest_label`, `interest_reasoning`

#### `rank_candidates(enriched) → list` (Local, no API)
- `final_score = match_score × 0.6 + interest_score × 0.4`
- Sorts descending, assigns rank 1–5

### 4.7 Total API Calls Per Request

| Stage            | Calls | Notes                          |
|------------------|-------|--------------------------------|
| JD Parse         | 1     | Single prompt                  |
| Candidate Scoring| 1     | All 20 in one batch prompt     |
| Outreach Sim     | 1     | All 5 in one batch prompt      |
| **Total**        | **3** | Fits 15 RPM with 57s headroom |

---

## 5. Scoring Formulas (fully transparent)

```
det_score    = skills_coverage_pct × 0.60 − min(exp_gap × 10, 30) + (15 if domain_match else 0)
match_score  = round(llm_holistic × 0.5 + det_score × 0.5)
final_score  = match_score × 0.60 + interest_score × 0.40
```

- **Match Score (60% of final)**: Blended LLM judgment + deterministic formula
- **Interest Score (40% of final)**: Simulated candidate engagement likelihood
- **Interest Labels**: Hot (70-100), Warm (40-69), Cold (0-39)

Both `llm_match_score` and `det_match_score` are exposed in the API response for full auditability.

---

## 6. Frontend (`frontend/index.html`)

### Design System
- **Theme:** Dark bg `#0a0a0f`, blue `#2563eb`, cyan `#06b6d4`
- **Fonts:** DM Sans (body), Space Grotesk (headings/scores) via Google Fonts CDN

### UI Components
- **Demo banner** (FIX-F1): Blue info bar when viewing pre-computed results
- **Pipeline meta bar** (FIX-F3): Scored count, API calls, wall time, weight split
- **Score breakdown** (FIX-F4): Skills coverage %, exp gap, domain match, LLM vs Det scores
- **Availability badge** (FIX-F5): Red "not looking" badge on candidate cards
- **Failed scoring warning** (FIX-F2): Amber banner if any candidates couldn't be scored
- **Empty state**: Clear message if no candidates could be scored
- **Score bars**: Animated fill transitions (800ms cubic-bezier)
- **Chat UI**: Expandable conversation with recruiter/candidate bubble styling

---

## 7. Security & Hardening

| Measure                | Implementation                                           |
|------------------------|----------------------------------------------------------|
| Prompt injection       | `sanitise_jd()` strips suspicious patterns; `<jd>/<candidate>` delimiters |
| PII stripping          | `SCORER_FIELDS` / `OUTREACH_FIELDS` whitelist before any LLM call |
| Score clamping         | All scores `max(0, min(100, int(...)))` in Python        |
| Availability enforcement | Dual-layer: prompt constraint + post-parse Python clamp |
| Input validation       | Pydantic `field_validator`, min 50 chars                 |
| Startup guard          | `RuntimeError` if `GEMINI_API_KEY` not set               |
| Concurrency guard      | `asyncio.Semaphore(2)` on all Gemini calls               |
| Timeout                | `asyncio.wait_for(..., timeout=45.0)` per call           |

---

## 8. Known Issues & Mitigations

| Issue                        | Mitigation                                              |
|------------------------------|---------------------------------------------------------|
| Flash Lite broken JSON mode  | Dual-mode wrapper: JSON mode → text mode fallback       |
| Free-tier 15 RPM limit       | 3-call batch architecture (57s headroom per minute)     |
| Free-tier 20 RPD on 2.5-flash | Using `flash-lite-latest` instead                      |
| Windows cp1252 encoding      | ASCII `[WARNING]` instead of emoji in console           |
| Gemini quota exhaustion      | Demo mode (`/api/demo`) as zero-API-cost safety net     |

---

## 9. How to Run

```bash
cd talentpulse
pip install -r requirements.txt
# Edit .env with your GEMINI_API_KEY
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/frontend/index.html
```
