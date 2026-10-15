# LeakGuard Gateway

AI security gateway for a 24-hour hackathon (problem CC6-AI03): sits between users
and an LLM, inspects prompts and responses, masks or blocks sensitive data and
prompt injection, and enforces role-based document access.

## Stack
Python, FastAPI (gateway), Streamlit (UI), SQLite (audit log), Chroma (RAG, Round 2).
Development is on Windows with PowerShell.

## Team ownership — do not edit other people's files without asking
- gateway/main.py, policy.py, policy.yaml, rag/ : Meihul
- gateway/detectors/ : Swasti
- ui/ : Anjali
- data/docs/, tests/prompts.csv, slides : Kashish

## Contracts
Every detector: function(text: str) -> list[dict], each dict:
{"category", "span", "start", "end", "severity", "confidence", "detector"}

POST /chat request:  {"user", "role", "prompt"}
POST /chat response: {"response", "action": "allow|mask|block", "findings", "masked_prompt"}

Detectors run on NORMALIZED text; masking uses the same normalized text.

## Rules
- Never commit .env or any real secret.
- Never log raw sensitive values; the audit log stores masked snippets only.
- Roles: intern, manager, hr.