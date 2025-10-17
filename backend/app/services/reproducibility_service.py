"""Reproducibility Service

Builds a lightweight reproducibility record for each run.
Includes environment versions, dataset fingerprint, seeds, and key decisions.

Contract
- build_reproducibility(job_id, manifest, eda) -> dict
  - Persists minimal info helpful for reruns and audits

Design
- Avoids heavy operations: hashes only first couple MB of dataset file when possible
- Graceful fallbacks if path is remote or unavailable
"""

from __future__ import annotations
from typing import Any, Dict

import sys
import hashlib
import platform
from pathlib import Path


def _file_fingerprint(path: Path, max_bytes: int = 2 * 1024 * 1024) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        st = path.stat()
        info.update(
            {
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "name": path.name,
                "suffix": path.suffix,
            }
        )
        h = hashlib.sha1()
        with path.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        info["sha1_head"] = h.hexdigest()
    except Exception:
        pass
    return info


def build_reproducibility(
    job_id: str, manifest: Dict[str, Any], eda: Dict[str, Any]
) -> Dict[str, Any]:
    # Environment
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import sklearn  # noqa: F401

        np_ver = __import__("numpy").__version__
        pd_ver = __import__("pandas").__version__
        sk_ver = __import__("sklearn").__version__
    except Exception:
        np_ver = pd_ver = sk_ver = None
    env = {
        "python": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "numpy": np_ver,
        "pandas": pd_ver,
        "sklearn": sk_ver,
    }

    # Dataset fingerprint (best-effort)
    ds = {}
    try:
        dataset_path = manifest.get("dataset_path")
        dspath = Path(dataset_path) if dataset_path else None
        if dspath and dspath.exists():
            ds = _file_fingerprint(dspath)
            ds["path_basename"] = dspath.name
    except Exception:
        pass

    # Decisions and framing snapshot
    decisions = ((manifest or {}).get("router_plan") or {}).get("decisions") or {}
    framing = (manifest or {}).get("framing") or {}

    # Seeds and CV (common defaults recorded)
    seeds = {"random_state": 42}
    cv = {"cv_folds": 5}

    repro = {
        "env": env,
        "dataset": ds,
        "decisions": decisions,
        "framing": framing,
        "seeds": seeds,
        "cv": cv,
    }
    return repro
