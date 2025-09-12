# Backend API Reference

Base URL: http://localhost:8000

Authentication: none (local development). CORS allows http://localhost:3000 by default.

## Upload
POST /upload

- Request: multipart/form-data with field `file`
- Response 200 (JSON):
```
{
  "job_id": "<hex>",
  "dataset_path": "<server path>",
  "file_format": "csv|tsv|excel",
  "sheet_names": ["Sheet1", ...] // only for Excel; may be null
}
```
- Errors:
  - 400: unsupported file type
  - 413: file too large

## Analyze
POST /analyze

- Request (JSON):
```
{
  "job_id": "<optional, to resume>",
  "dataset_path": "<from /upload>",
  "file_format": "csv|tsv|excel|auto",
  "nl_description": "freeform context",
  "question": "freeform; e.g., 'Classify … target=Survived'",
  "sheet_name": null,
  "delimiter": ","
}
```
- Response 200 (JSON):
```
{ "job_id": "<hex>" }
```
- Behavior: enqueues a background run of the pipeline for the job.

## Status
GET /status/{job_id}

- Response 200 (JSON):
```
{
  "job_id": "<hex>",
  "status": "RUNNING|COMPLETED|FAILED|CANCELLED|UPLOADED|CREATED",
  "progress": 0-100,
  "stage": "eda|modeling|report|qa|ingest",
  "messages": [{"role": "assistant|system", "content": "..."}]
}
```
- 404 if job not found

## Result
GET /result/{job_id}

- Response 200 (JSON):
```
{
  "phase": "eda-only" | "full",
  "eda": { ... },
  "modeling": { ... },
  "explain": { ... },
  "qa": { "issues": [] },
  "report_html": "<html>…</html>" // present when reporting completes
}
```
- Note: For long jobs, EDA-only results may be available early.

## Clarify
POST /clarify

- Request (JSON):
```
{ "job_id": "<hex>", "message": "target=<col> [metric=<name>]" }
```
- Response 200 (JSON): `{ "ok": true }`
- Behavior: updates manifest (target/metric) and resumes modeling via the queue

## Cancel
POST /cancel/{job_id}

- Response 200 (JSON): `{ "ok": true }`
- Behavior: sets a cancel flag; pipeline exits at safe boundaries

## Health
GET /health → `{ "ok": true }`

---

## Data contracts (simplified)

### EDA (excerpt)
```
{
  "shape": {"rows": 1000, "cols": 12},
  "columns": ["age", "fare", ...],
  "missing": {"age": {"count": 10, "pct": 1.0}, ...},
  "dtypes": {"age": "int64", ...},
  "plots": { "missingness": "/static/jobs/<id>/plots/missingness.png", "histograms": [...], "categoricals": [...] }
}
```

### Modeling (excerpt)
```
{
  "task": "classification|regression",
  "leaderboard": [{"name": "rf_clf", "f1": 0.81, "cv_mean": 0.79, ...}, ...],
  "best": {"name": "rf_clf", "f1": 0.82, "acc": 0.85, "tuned_threshold": 0.47},
  "features": {"numeric": 5, "categorical": 3},
  "selected_tools": ["logreg", "rf_clf", "hgb_clf"],
  "explain": { ... }
}
```

### Explain (excerpt)
```
{
  "importances": [0.15, 0.12, ...],
  "roc": "/static/jobs/<id>/plots/roc.png",
  "pr": "/static/jobs/<id>/plots/pr.png",
  "pdp": ["/static/jobs/<id>/plots/pdp_age.png", ...]
}
```

## Sample data (demo)
POST /sample

- Response 200 (JSON): `{ "job_id": "<hex>" }`
- Behavior: creates a tiny built-in CSV under the job and enqueues analysis (classification on Survived)


### Report
- `report_html` is a self-contained HTML string.
- When REPORT_JSON_FIRST is enabled and OpenAI available, a JSON report is validated and rendered deterministically.


