# TalentPulse Debugging & Architecture Report

## Problem Statement
The TalentPulse AI pipeline is consistently failing to parse the Job Description (falling back to a generic "Software Engineer" placeholder) and failing to score all 20 candidates (resulting in `Match Score 0` and `20 scoring failures` showing in the metadata bar). 

Despite updating to a freshly generated Gemini API key, the pipeline is being blocked by strict free-tier quota limits across multiple Gemini models.

## Architectural Changes Implemented
To harden the pipeline for production and attempt to bypass these rate limits, the following structural changes were implemented in `main.py` and `frontend/index.html`:

1. **Async Refactoring**: Migrated the entire pipeline to native Python `asyncio`. Used `asyncio.to_thread` for the synchronous Gemini SDK calls and replaced all blocking `time.sleep` calls with `await asyncio.sleep()`.
2. **Backoff & Timeouts**: Implemented a robust `gemini_call_with_retry` wrapper.
   - Enforces a `call_timeout=30.0` seconds via `asyncio.wait_for()`.
   - Uses exponential backoff on `429 Too Many Requests` (4s -> 8s -> 16s -> 32s).
   - `max_retries` increased to 5.
3. **Ghost Process Cleanup**: Fixed an issue where older orphaned `uvicorn` instances were silently hogging port 8000, preventing new code changes (and the new API key in `.env`) from taking effect.
4. **Model Rotation**: 
   - `gemini-1.5-flash`: Deprecated/unavailable on the new API key (`NotFound` error).
   - `gemini-2.0-flash`: Hard-capped at exactly `0` requests per minute on the new key's free tier (`ResourceExhausted` limit: 0).
   - `gemini-2.5-flash`: Hard-capped at `20` requests **per day** on the free tier. Since our pipeline requires 26 API calls per run, it immediately exhausted the daily quota on the first attempt.
   - `gemini-flash-lite-latest`: Has a limit of `15 RPM`. We attempted to use this with 4.1-second artificial delays between calls, but it is still sporadically failing and returning fallback structures.
5. **JSON Extraction Hardening**: Upgraded `extract_json()` to strictly validate responses using an `expected_keys` array (e.g., enforcing that the JD parse returns `"role_title"` and `"required_skills"`). If the model hallucinates or the rate-limit interrupts it, it safely catches the error and returns a predefined fallback dictionary to prevent frontend crashes.

## What Claude Needs to Suggest
Given that a single end-to-end "Scout Candidates" action fundamentally requires **26 API calls** (1 for JD + 20 for candidates + 5 for outreach), please advise on:

1. **Quota Workarounds**: Are there better ways to structure the Gemini SDK calls or prompt chaining so we don't hit the 15 RPM / 20 RPD free tier limits? 
2. **Batching**: How can we rewrite the `_score_single_candidate` logic to score all 20 candidates in a single Gemini prompt instead of 20 separate API calls, drastically reducing the RPM footprint?
3. **Silent Failures**: The `extract_json` block is safely catching exceptions, but the `gemini-flash-lite-latest` model seems to be struggling to return the rigidly requested JSON format, triggering the fallback constantly. Should we switch to the structured `response_schema` feature natively provided by the newer Gemini SDK instead of regex parsing?
