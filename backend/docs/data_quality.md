# Data Quality & Leakage Guardrails

Conservative, fast checks executed before modeling to surface common issues.

What is checked
- Target viability
  - Classification: class prevalence (binary) or minimal per-class counts (multi-class)
  - Regression: non-zero variance
- Identifier columns
  - Near-unique columns (>=95% unique) or ID-like names with high uniqueness
- Near-perfect predictors (leakage signals)
  - Binary: numeric features with ROC AUC ~1, or categorical levels that are nearly pure to one class covering most rows
  - Regression: numeric features with Pearson correlation ~1
- Timeseries note
  - If router selected time split and time columns exist, warn to avoid future information in features

Outputs
- JSON written to job_dir/data_quality.json and persisted in manifest["data_quality"], e.g.
```
{
  "issues": [ {"id": "identifier_columns", "severity": "warn", "detail": "..." } ],
  "recommendations": ["..."],
  "summary": "identifier_columns; near_perfect_predictors"
}
```

Design principles
- Cheap and conservative: avoid long scans and keep false positives low
- Non-blocking: surfaced as guidance; does not stop the pipeline by default
- Evolvable: implemented behind a service facade for future expansion

