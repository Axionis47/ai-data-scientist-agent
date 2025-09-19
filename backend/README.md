# ai-data-scientist-agent (Backend)

## Quick start
1. Create a virtualenv and install dependencies:
   - python3 -m venv .venv
   - source .venv/bin/activate
   - pip install -U pip && pip install -e . -r requirements-dev.txt
2. Run the server:
   - uvicorn app.main:app --reload --port 8000

## Environment
- Python 3.10+
- FastAPI, Uvicorn, Pandas, scikit-learn, openpyxl

## API
- POST /upload: multipart file upload, returns {job_id, dataset_path, file_format, sheet_names?}
- POST /analyze: JSON {job_id?, dataset_path, file_format?, nl_description, question, sheet_name?, delimiter?}
- POST /clarify: JSON {job_id, message}
- GET /status/{job_id}
- GET /result/{job_id}
- POST /cancel/{job_id}
- GET /health

## Storage
- data/jobs/{job_id}/original/<file>
- manifest.json, result.json, eda.json, modeling.json, plots/, telemetry.jsonl

## Friendly notes (for first-time AI agent builders)
- It is okay to start small. Try with a tiny CSV and use only /upload and /analyze first.
- If OpenAI key is missing, reporting will use a deterministic fallback.
- Logs in backend/data/jobs/{job_id}/telemetry.jsonl can help when something looks odd.

## Further docs
- docs/ARCHITECTURE.md — modules, stages, state/queue, artefacts
- docs/OPERATIONS.md — env vars, endpoints, resumability, benchmarks, troubleshooting
- docs/API.md — endpoints and contracts
