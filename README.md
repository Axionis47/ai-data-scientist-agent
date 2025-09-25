# ai-data-scientist-agent

[![CI](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml)

This project is an AI Data Scientist agent. You give it a dataset and a question, and it will do quick EDA, try a simple ML model (if useful), and produce a neat HTML report. It is built with FastAPI, Pandas and scikit-learn. CI is set up, and the backend is prepared for future serverless (GCP) options.

If you are new to AI agents: no stress. Start the backend, use the API to upload a CSV/Excel file, and then ask the agent to analyse it. The defaults are conservative.

## Quick start (backend)

- cd backend
- python3 -m venv .venv && source .venv/bin/activate
- pip install -U pip && pip install -e . -r requirements-dev.txt
- uvicorn app.main:app --reload --port 8000

## API (backend)
- POST /upload: upload CSV/TSV/Excel
- POST /analyze: start a background job on the uploaded file
- GET /status/{job_id}: check progress
- GET /result/{job_id}: fetch results JSON (includes report_html)
- POST /clarify: send clarification (for example, target=y)
- GET /health: health check

## CI/CD
- GitHub Actions: CI runs lint, type checks, security and tests on PRs and pushes
- Coverage uploaded to Codecov (optional)
- Dockerfile provided for backend
- CD workflow builds and pushes an image to GHCR (latest + sha), and runs a Trivy image scan

## OpenAI setup
- Local (backend/.env):
  - Create backend/.env (gitignored) with:
    - OPENAI_API_KEY=sk-...
  - Start backend: uvicorn app.main:app --reload --port 8000
- Docker Compose:
  - Create a .env in repo root (same folder as docker-compose.yml) with:
    - OPENAI_API_KEY=sk-...
  - Run: docker compose up -d backend
- GitHub Actions (docker-smoke):
  - Add repository secret OPENAI_API_KEY and re-run the workflow

Note: If you do not have a key, the app will fall back to a non-OpenAI path for report generation.

## Serverless-ready (GCP)
- Adapters are scaffolded for Storage (Local/GCS), JobStore (Memory/Firestore), JobQueue (Local/PubSub)
- Local development remains unchanged; a future PR will wire adapters behind environment flags

## Frontend (optional)
- A small Next.js app lives in frontend/ (landing page and results)

## License
MIT
