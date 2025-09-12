from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
BENCH_DIR = ROOT / "bench"

API = os.getenv("BENCH_API", "http://localhost:8000")


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _upload_csv(client, csv_bytes: bytes, name: str) -> Dict[str, Any]:
    files = {"file": (name, io.BytesIO(csv_bytes), "text/csv")}
    r = client.post(f"{API}/upload", files=files)
    r.raise_for_status()
    return r.json()


def run_one(client, ds: Dict[str, Any]) -> Dict[str, Any]:
    # Load dataset (CSV only for seed)
    if ds["path"].startswith("http"):
        csv_bytes = requests.get(ds["path"], timeout=30).content
    else:
        csv_bytes = Path(ds["path"]).read_bytes()
    info = _upload_csv(client, csv_bytes, name=f"{ds['id']}.csv")
    job_id = info["job_id"]
    dataset_path = info["dataset_path"]

    # Submit analyze
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": f"benchmark {ds['id']}",
        "question": ("classify" if ds["task"] == "classification" else "regression"),
        "delimiter": ",",
    }
    r = client.post(f"{API}/analyze", json=body)
    r.raise_for_status()

    # Poll until done or clarify
    deadline = time.time() + 120
    stage = None
    while time.time() < deadline:
        st = client.get(f"{API}/status/{job_id}").json()
        stage = st.get("stage")
        if st.get("status") in ("COMPLETED", "FAILED"):
            break
        if stage == "clarify" and ds.get("target"):
            # Provide clarification
            msg = f"target={ds['target']}"
            client.post(f"{API}/clarify", json={"job_id": job_id, "message": msg})
        time.sleep(0.5)

    # Collect result
    res = client.get(f"{API}/result/{job_id}")
    if res.status_code == 200:
        result = res.json()
        metrics = (result.get("modeling") or {}).get("metrics") or {}
        primary = metrics.get("primary")
        return {
            "job_id": job_id,
            "dataset": ds["id"],
            "stage": stage,
            "status": st.get("status"),
            "metric_primary": primary,
            "durations_ms": (result.get("timings") or {}),
        }
    return {"job_id": job_id, "dataset": ds["id"], "stage": stage, "status": st.get("status")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BENCH_DIR / "bench.csv"))
    parser.add_argument("--datasets", default=str(BENCH_DIR / "datasets.yaml"))
    args = parser.parse_args()

    with open(args.datasets, "r") as f:
        cfg = yaml.safe_load(f)
    datasets: List[Dict[str, Any]] = cfg.get("datasets", [])

    # simple client using requests
    class Client:
        def post(self, url, **kw):
            return requests.post(url, **kw)
        def get(self, url, **kw):
            return requests.get(url, **kw)

    rows = []
    for ds in datasets:
        try:
            rows.append(run_one(Client(), ds))
        except Exception as e:
            rows.append({"dataset": ds["id"], "error": str(e)})

    # write CSV
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*[set(r.keys()) for r in rows if isinstance(r, dict)]))
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()

