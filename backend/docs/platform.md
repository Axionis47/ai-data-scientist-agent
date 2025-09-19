# Platform adapters

- app/platform/jobstore.py
  - Disk-based helpers: job_dir, read_json, write_json, mark_done, exists
- app/platform/statemachine.py
  - Validates and records stage transitions; computes durations_ms and stage_starts
- app/platform/queue_runner.py
  - Local thread-based queue with concurrency cap; used by the API to run jobs

Tip
- In local development, these defaults are fine. Later, you can swap for cloud equivalents.
