ai-data-scientist-agent (Backend)

Quick Start

1. Create venv and install deps:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .

2. Run server:
   uvicorn app.main:app --reload --port 8000

Environment
- Python 3.10+
- FastAPI, Uvicorn, Pandas, openpyxl

API
- POST /upload: multipart file upload, returns {job_id, dataset_path, file_format, sheet_names?}
- POST /analyze: JSON {job_id?, dataset_path, file_format?, nl_description, question, sheet_name?, delimiter?}
- GET /status/{job_id}
- GET /result/{job_id}
- POST /clarify

Storage
- backend/data/jobs/{job_id}/original/<file>
- manifest.json, result.json

