# Pipeline Orchestration (app/pipeline/run.py)

Purpose
- Orchestrates stages: ingest → eda → clarify (optional) → modeling → report → qa → done
- Resumable with .done markers and artifact checks

Key responsibilities
- Reads manifest, loads dataset, calls EDA, handles clarify gating
- Calls modeling and reporting; writes result.json and report.done
- Records timings and decisions; appends telemetry.jsonl

Artifacts
- data/jobs/{job_id}/
  - manifest.json(.done), eda.json(.done), modeling.json(.done), result.json, report.done, plots/*.png, telemetry.jsonl

Notes
- Enforces stage timeouts if configured
- Uses state machine to guard transitions and compute durations

