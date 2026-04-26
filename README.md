# ⚡ TalentPulse — AI-Powered Talent Scouting & Engagement Agent

TalentPulse is an AI-powered recruiting tool that takes a raw Job Description and automatically parses it, batch-scores 20 candidates, simulates recruiter outreach conversations, and returns a ranked shortlist — all in under 20 seconds using just 3 API calls.

Built for the **Catalyst Hackathon** by **Deccan AI Experts**.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Recruiter   │────▶│   Frontend   │────▶│   FastAPI        │────▶│  Gemini API  │
│  (Browser)   │◀────│  index.html  │◀────│   Backend        │◀────│  Flash Lite  │
└─────────────┘     └──────────────┘     └────────┬────────┘     └──────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ candidates.json │
                                          │  (20 profiles)  │
                                          └─────────────────┘
```

**Pipeline Flow (3 API calls total):**
1. **JD Parsing** (Call 1) → Gemini extracts structured fields (skills, experience, domain)
2. **Batch Scoring** (Call 2) → All 20 candidates scored in one prompt, blended with deterministic formula
3. **Batch Outreach** (Call 3) → Top 5 candidates get simulated 4-turn conversations in one prompt
4. **Final Ranking** (Local) → `final_score = match_score × 0.6 + interest_score × 0.4`

---

## 📊 Scoring Logic

| Score | Range | Description |
|-------|-------|-------------|
| **Match Score** | 0–100 | How well the candidate's skills/experience align with the JD |
| **Interest Score** | 0–100 | Simulated likelihood the candidate would engage (Hot/Warm/Cold) |
| **Final Score** | 0–100 | `match_score × 0.6 + interest_score × 0.4` |

**Why 60/40?** Skills fit is the primary indicator of success, but a highly-skilled candidate who isn't interested provides no value. The 40% interest weight ensures the shortlist prioritizes actionable leads.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| AI | Google Gemini Flash Lite (`google-generativeai`) |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Data | Static JSON (20 candidates) |
| Config | `.env` with `python-dotenv` |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone <repo-url>
cd talentpulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API key
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a free API key at: https://aistudio.google.com/apikey

### 4. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend
Open `frontend/index.html` in your browser, or navigate to:
```
http://localhost:8000/frontend/index.html
```

---

## 📡 API Reference

### `GET /health`
Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

### `GET /api/candidates`
Returns all 20 candidates from the database.

**Response:** Array of candidate objects.

### `POST /api/analyze`
Main pipeline endpoint. Parses JD, matches candidates, simulates outreach, returns ranked shortlist.

**Request:**
```json
{
  "job_description": "Senior Backend Engineer at FinStack Technologies..."
}
```

**Response:**
```json
{
  "parsed_jd": {
    "role_title": "Senior Backend Engineer",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", ...],
    "preferred_skills": ["Kafka", "Kubernetes"],
    "min_experience_years": 4,
    "education_requirement": "B.Tech/B.E.",
    "role_type": "full-time",
    "domain": "backend",
    "key_responsibilities": [...]
  },
  "shortlist": [
    {
      "rank": 1,
      "candidate": { ... },
      "match_score": 95,
      "interest_score": 78,
      "final_score": 88.2,
      "matched_skills": [...],
      "missing_skills": [...],
      "match_explanation": "...",
      "conversation": [...],
      "interest_label": "Hot",
      "interest_reasoning": "..."
    }
  ],
  "pipeline_summary": "..."
}
```

---

## 📁 Sample Input/Output

- **Sample JD:** [`sample_inputs/sample_jd.txt`](sample_inputs/sample_jd.txt)
- **Sample Output:** [`sample_outputs/sample_output.json`](sample_outputs/sample_output.json)

⚡ To demo without API calls: click "Try Demo" — uses pre-computed results.

---

## 🧮 Scoring Formula (Explainability)

All scoring logic is transparent and auditable:

```text
det_score   = skills_coverage_pct × 0.60
            − min(experience_gap_years × 10, 30)
            + (15 if domain_match else 0)

match_score = round(llm_holistic × 0.5 + det_score × 0.5)

final_score = match_score × 0.60 + interest_score × 0.40
```

Both `llm_match_score` and `det_match_score` are returned in the API response
for independent verification. The deterministic component is computed locally
in Python with no LLM involvement.

---

## ⚠️ Known Limitations

- **Free tier quota:** Uses `gemini-flash-lite-latest` (15 RPM). Under quota
  pressure, some candidates may not be scored. The frontend shows a warning
  and returns the best available partial results.
- **Occasional JSON fallback:** The SDK intermittently returns malformed JSON
  in JSON mode. The pipeline automatically retries in text mode — this adds
  ~2–4 seconds on affected calls but does not crash the pipeline.
- **Demo mode:** `/api/demo` serves pre-computed results with zero API calls.
  Use this if Gemini quota is exhausted during judging.

---

## 🧠 Architecture Decisions

- **Gemini Flash Lite** — Chosen for speed and generous free-tier quotas (15 RPM). Perfect for a sub-20s end-to-end demo utilizing prompt batching.
- **Top-of-Funnel Truncation** — The pipeline scores all 20 candidates on skills but only runs the 4-turn outreach simulation on the Top 5. This explicitly mirrors real-world recruiter workflows (you only engage the most technically qualified leads) and prevents highly interested but totally unqualified candidates from polluting the final shortlist.
- **60/40 Weighting** — Skills match is necessary but not sufficient. A candidate who won't respond is a wasted outreach. The 40% interest weight balances fit vs. reachability.
- **Simulated Outreach** — Real outreach takes days. Simulation gives recruiters a preview of likely candidate responsiveness, enabling smarter prioritization before spending time on actual outreach.
- **Single-file Frontend** — Zero build step, zero dependencies. Open the file and it works. Perfect for a hackathon demo.
- **Batch API Calls** — We use the 1M token context window to process all 20 candidates and all 5 outreach simulations simultaneously. This consolidates 26 API calls down to just 3 calls, completely avoiding free-tier rate limits.

---

## 🏆 Hackathon Submission

**Event:** Catalyst Hackathon  
**Team:** Deccan AI Experts  
**Category:** AI-Powered Talent Scouting & Engagement Agent
