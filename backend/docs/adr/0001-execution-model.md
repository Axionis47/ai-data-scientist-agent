# ADR 0001: Execution Model and Resumability

## Context
We need reliable background execution, progress visibility, and deterministic resumability to support large files and long-running jobs.

## Decision
- Use a local thread-backed job queue (QueueRunner + LocalThreadQueue) with a concurrency cap (MAX_CONCURRENT_JOBS)
- Represent pipeline lifecycle as a simple state machine with validated transitions and a timeline
- Write stage artefacts and .done markers (eda/modeling/report) to support skip-on-resume
- Enforce per-stage timeouts (EDA/MODEL/REPORT) and allow cancellation via flag checked at heavy boundaries

## Consequences
- Local dev is simple and robust; can later swap queue/store with cloud backends
- Artifacts provide observability and recovery; small disk footprint tradeoff
- Timeouts avoid hangs; cancellation gives responsive UX

