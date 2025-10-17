import json
import re
from typing import Dict, Any
from .config import JOBS_DIR

_target_re = re.compile(r"target\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_metric_re = re.compile(r"metric\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


def apply_clarification(job_id: str, text: str, manifest: Dict[str, Any]):
    target_m = _target_re.search(text or "")
    if target_m:
        manifest.setdefault("framing", {})["target"] = target_m.group(1)
    metric_m = _metric_re.search(text or "")
    if metric_m:
        manifest.setdefault("framing", {})["metric"] = metric_m.group(1)
    # persist manifest change
    job_dir = JOBS_DIR / job_id
    (job_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # write checkpoint
    try:
        (job_dir / "manifest.done").write_text("ok", encoding="utf-8")
    except Exception:
        pass
