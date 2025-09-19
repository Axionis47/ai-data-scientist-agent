# Pipeline Orchestration (app/pipeline/run.py)

Purpose
- Orchestrates stages: ingest → eda → clarify (optional) → modeling → report → qa → done
- Resumable with .done markers and artefact checks

Key responsibilities
- Reads the manifest, loads the dataset, calls EDA, and handles clarify gating
- Calls modelling and reporting; writes result.json and report.done
- Records timings and decisions; appends telemetry.jsonl

Artefacts
- data/jobs/{job_id}/
  - manifest.json(.done), eda.json(.done), modeling.json(.done), result.json, report.done, plots/*.png, telemetry.jsonl

Notes
- Enforces stage timeouts if configured
- Uses the state machine to guard transitions and compute durations

