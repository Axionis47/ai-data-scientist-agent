# Reporting module (app/reporting/report.py)

Entry point
- reporting_expert(job_id, eda, modeling, explain) -> HTML string

Modes
- JSON-first (feature-flagged): requests structured JSON, validates it, and renders deterministic HTML
- HTML via LLM: prompts for clean HTML with inline CSS
- Fallback: deterministic HTML without LLM; includes a model card, risks/caveats, and anchors

Utilities
- _inline_img: optional base64 inlining of images (REPORT_INLINE_ASSETS)
- _span: OpenTelemetry span helper (no-op if OTEL not installed)

Inputs
- EDA: summary fields and plots
- Modeling: best metrics, task, features, selected tools, and explain assets
- Explain: ROC/PR, PDPs, diagnostics

Notes for newcomers
- If the OpenAI key is not set, the module will fall back to the deterministic path.
- You can disable inline assets if page size becomes too large.
