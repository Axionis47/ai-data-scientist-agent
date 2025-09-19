# Core modules

- app/core/config.py
  - Central repository of env-config and feature flags
- app/core/schemas.py
  - Lightweight validators for EDA/modeling/report JSON
- app/core/logs.py
  - Human-readable decision logger used throughout
- app/core/telemetry.py
  - Appends structured entries to telemetry.jsonl per job

Notes for beginners
- If a flag name is confusing, search for it in the codebase to see usage.
