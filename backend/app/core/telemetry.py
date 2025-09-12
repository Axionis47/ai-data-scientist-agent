from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any

from .config import JOBS_DIR


def append_run_telemetry(job_id: str, payload: Dict[str, Any]) -> None:
    """Append a JSON line to telemetry file under the job folder.
    Non-fatal. Intended for quick local analysis.
    """
    try:
        tpath = JOBS_DIR / job_id / "telemetry.jsonl"
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        with tpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

