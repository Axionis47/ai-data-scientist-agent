# Backend files documentation

Below are key backend files and what each does, with inputs/outputs and key functions.

- app/main.py
  - Purpose: FastAPI app, routes, static serving, queue bootstrapping
  - Key endpoints: /upload, /analyze, /clarify, /status/{id}, /result/{id}, /cancel/{id}, /health
  - Inputs: HTTP requests; files via multipart; query/body JSON
  - Outputs: JSON responses; static files under /static/jobs/{id}

- app/pipeline/run.py
  - Purpose: Orchestrates the end-to-end pipeline per job
  - Inputs: manifest.json, uploaded file
  - Outputs: eda.json(.done), modeling.json(.done), result.json, report.done, plots/*.png, telemetry.jsonl
  - Key: resumable stages, timeouts, telemetry, model_decision logging

- app/platform/jobstore.py
  - Purpose: File-based job store helper
  - APIs: read_json, write_json, mark_done, exists, job_dir

- app/platform/statemachine.py
  - Purpose: Track stage transitions, enforce valid ordering, timings
  - Fields: job.timeline, durations_ms, stage_starts

- app/platform/queue_runner.py
  - Purpose: Lightweight thread-based queue with concurrency cap
  - Inputs: jobs scheduled by API; config MAX_CONCURRENT_JOBS

- app/eda/eda.py
  - Purpose: Compute EDA, detect dtypes, simple charts, warnings
  - Outputs: eda.json with shape, columns, dtype map, sample, charts

- app/modeling/pipeline.py
  - Purpose: Train baseline models (classification/regression), CV, explainability
  - Outputs: modeling.json; best model metrics; optional plots (ROC/PR, diagnostics)

- app/reporting/report.py
  - Purpose: Generate report_html via JSON-first LLM path or deterministic fallback
  - Flags: REPORT_JSON_FIRST, REPORT_INLINE_ASSETS
  - Utilities: _inline_img, _render_model_card, _span

- app/core/config.py
  - Purpose: Centralized configuration and feature flags
  - Important flags: MAX_* timeouts, SAFE_AUTO_ACTIONS, REPORT_* brand tokens, SHAP_*

- app/core/schemas.py
  - Purpose: Validate EDA, modeling, and report JSON structures
  - Notable: validate_report_json accepts optional model_card

- app/core/logs.py, app/core/telemetry.py
  - Purpose: Human-friendly decisions; machine-friendly telemetry JSONL

- app/explainers/*
  - Purpose: helpers to add curves and diagnostics; SHAP optional

Conventions
- All artifacts per job live under backend/data/jobs/{job_id}
- Each stage writes a .done file as a durable checkpoint
- Static routes expose only safe directories (no original uploads by default)

