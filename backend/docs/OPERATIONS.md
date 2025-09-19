# Operations Guide

## Quick start
- python3 -m venv .venv && source .venv/bin/activate
- pip install -U pip && pip install -e . -r requirements-dev.txt
- uvicorn app.main:app --reload --port 8000

## Environment variables
- Concurrency and timeouts
  - MAX_CONCURRENT_JOBS: default 1
  - EDA_TIMEOUT_S / MODEL_TIMEOUT_S / REPORT_TIMEOUT_S: default 0 (disabled)
- Reporting
  - REPORT_JSON_FIRST: false by default; when true, try JSON-first reporting (OpenAI key required)
  - Brand tokens: REPORT_PRIMARY/ACCENT/BG/SURFACE/TEXT/MUTED/OK/WARN/ERROR/FONT_FAMILY/LOGO_URL
- Modelling
  - CV_FOLDS, EARLY_STOP_SAMPLE, HGB_MIN_ROWS, SEARCH_TIME_BUDGET, SHAP_ENABLED, SHAP_MAX_ROWS
  - SAFE_AUTO_ACTIONS: when true, apply a conservative class_weight=balanced hint under imbalance
- Security
  - STATIC_EXPOSE_ORIGINAL: false by default; keep original/ private

## Endpoints
- POST /upload → returns {job_id, dataset_path}
- POST /analyze → start a job
- POST /clarify → resume modelling (for example, target=y)
- GET /status/{job_id}
- GET /result/{job_id}
- POST /cancel/{job_id}
- GET /health

## Artefacts
- data/jobs/{job_id}/
  - manifest.json, manifest.done
  - eda.json, eda.done
  - modeling.json, modeling.done
  - result.json, report.done, (report.html optional)
  - critique.json (optional)
  - telemetry.jsonl
  - plots/*.png

## Resumability and recovery
- If eda.json + eda.done exist and validate → skip EDA on restart
- If modeling.json + modeling.done exist and have metrics → skip heavy retrain
- If report.done exists and result.json has report_html → treat reporting as complete
- Use /cancel to stop long jobs; pipeline exits between heavy steps

## Benchmark harness
- Run: python backend/bench/run_bench.py --out backend/bench/bench.csv
- Report: python backend/bench/report.py --csv backend/bench/bench.csv --out backend/bench/bench.html --thresholds backend/bench/thresholds.yaml

## Troubleshooting
- result.json missing → job still running or failed; check /status and telemetry.jsonl
- OpenAI unavailable → reporting falls back to the deterministic template
- Large CSVs slow → increase LARGE_FILE_MB and ensure chunked EDA is active
- Many categories → TopKCategorical reduces cardinality; adjust k in code if needed

## Testing
- Run all tests: pytest -q
- Focused runs:
  - test_api_flow.py::test_api_upload_analyze_clarify_and_result
  - test_clarify_and_resume.py::test_clarify_gating_and_resume
  - test_modeling_explain_logs.py::test_modeling_explain_and_logs
