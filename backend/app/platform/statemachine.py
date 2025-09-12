from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Simple state machine for pipeline stages
STAGES = (
    "ingest",
    "eda",
    "clarify",
    "modeling",
    "report",
    "qa",
    "done",
    "error",
    "cancelled",
)

ALLOWED_TRANSITIONS = {
    None: {"ingest", "eda"},
    "ingest": {"eda", "error", "cancelled"},
    "eda": {"clarify", "modeling", "error", "cancelled"},
    "clarify": {"modeling", "error", "cancelled"},
    "modeling": {"report", "error", "cancelled"},
    "report": {"qa", "error", "cancelled"},
    "qa": {"done", "error", "cancelled"},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def transition_stage(
    job_store, job_id: str, new_stage: str, extra_patch: Optional[Dict[str, Any]] = None
) -> None:
    """Validate and perform a stage transition. Records a timestamp per stage in job['timeline'].
    If transition is illegal, we still set the stage (to avoid hard failures), but add a 'illegal_transition' note.
    """
    job = job_store.get(job_id) or {}
    prev = job.get("stage")
    # Validate
    legal = new_stage in (ALLOWED_TRANSITIONS.get(prev) or set())
    timeline: Dict[str, str] = dict(job.get("timeline") or {})
    timeline[new_stage] = _now_iso()
    patch: Dict[str, Any] = {"stage": new_stage, "timeline": timeline}
    if not legal:
        notes = list(job.get("notes") or [])
        notes.append(
            {
                "type": "illegal_transition",
                "from": prev,
                "to": new_stage,
                "at": _now_iso(),
            }
        )
        patch["notes"] = notes
    if extra_patch:
        patch.update(extra_patch)
    job_store.update(job_id, patch)
