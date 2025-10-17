"""Service facade layer

Thin wrappers that expose stable, testable entry points for core subsystems
(EDA, modeling, router, reporting, critique). These deliberately re-export
functions from their underlying modules to:
- Provide a consistent interface boundary for the pipeline/orchestrator
- Enable future process isolation or microservice migration without touching callers
- Centralize documentation and contracts for each subsystem

Import from this package rather than from app.eda/app.modeling/etc in orchestration code.
"""
