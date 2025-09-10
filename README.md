# ai-data-scientist-agent

[![CI](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml)

An AI agent that ingests a dataset, runs fast EDA, optionally builds a simple ML model, and generates a clean HTML report. Built with FastAPI + Pandas + scikit-learn. Ready for CI and prepared for a future serverless (GCP) deployment.

## Quick Start (backend)

- cd backend
- python3 -m venv .venv && source .venv/bin/activate
- pip install -U pip && pip install -e . -r requirements-dev.txt
- uvicorn app.main:app --reload --port 8000

## API (backend)
- POST /upload: upload CSV/TSV/Excel
- POST /analyze: start a background job on the uploaded file
- GET /status/{job_id}: check progress
- GET /result/{job_id}: fetch results JSON (includes report_html)
- POST /clarify: send clarification (e.g., target=y)
- GET /health: health check

## CI/CD
- GitHub Actions: CI runs lint/type/security/tests on PRs and pushes
- Coverage uploaded to Codecov (optional dashboard)
- Dockerfile provided for backend
- CD workflow builds/pushes image to GHCR (latest + sha), with Trivy image scan

## Serverless-ready (GCP)
- Adapters scaffolded for Storage (Local/GCS), JobStore (Memory/Firestore), JobQueue (Local/PubSub)
- Local dev remains unchanged; future PR will wire adapters behind env flags

## Frontend (optional)
- Next.js app in frontend/ (landing page etc.)

## License
MIT

