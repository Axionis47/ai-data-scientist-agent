# ai-data-scientist-agent (Backend)

## Quick Start
1. Create venv and install deps:
   - python3 -m venv .venv
   - source .venv/bin/activate
   - pip install -U pip && pip install -e . -r requirements-dev.txt
2. Run server:
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

## Further docs
- docs/ARCHITECTURE.md — modules, stages, state/queue, artifacts
- docs/OPERATIONS.md — env vars, endpoints, resumability, benchmarks, troubleshooting

- docs/API.md — endpoints and contracts
