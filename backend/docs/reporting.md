# Reporting module (app/reporting/report.py)

Entry point
- reporting_expert(job_id, eda, modeling, explain) -> HTML string

Modes
- JSON-first (feature-flagged): requests structured JSON, validates, renders deterministic HTML
- HTML via LLM: prompts for clean HTML with inline CSS
- Fallback: deterministic HTML without LLM; includes model card, risks/caveats, anchors

Utilities
- _inline_img: optional base64 inlining of images (REPORT_INLINE_ASSETS)
- _span: OpenTelemetry span helper (no-op if OTEL not installed)

Inputs
- EDA: summary fields, plots
- Modeling: best metrics, task, features, selected tools, explain assets
- Explain: ROC/PR, PDPs, diagnostics

