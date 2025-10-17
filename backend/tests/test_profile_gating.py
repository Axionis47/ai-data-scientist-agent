from pathlib import Path
import io
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_profile_lean_skips_fairness(tmp_path: Path):
    client = TestClient(app)
    # Minimal dataset with a binary target
    df = pd.DataFrame(
        {
            "id": [f"A{i}" for i in range(30)],
            "age": list(range(30)),
            "city": ["X"] * 15 + ["Y"] * 15,
            "y": [0, 1] * 15,
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Upload
    files = {"file": ("lean.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    info = r.json()
    job_id = info["job_id"]
    dataset_path = info["dataset_path"]

    # Analyze with explicit lean profile; include target to avoid clarify path
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "classification target=y",
        "sheet_name": None,
        "delimiter": ",",
        "profile": "lean",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

    # Wait for pipeline to finish
    import time

    for _ in range(60):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            break
    assert st.get("status") == "COMPLETED"

    # Fetch result and verify fairness is skipped or empty in lean profile
    res = client.get(f"/result/{job_id}").json()
    fairness = res.get("fairness", None)
    assert (not fairness) or (
        "skipped_by_profile_lean" in (fairness.get("notes") or [])
    ), "fairness should be absent/empty or explicitly marked skipped in lean profile"


def test_profile_manifest_records_profile(tmp_path: Path):
    client = TestClient(app)
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"], "y": [0, 1, 0, 1]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("m.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/upload", files=files)
    job_id = r.json()["job_id"]
    dataset_path = r.json()["dataset_path"]

    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "classification target=y",
        "profile": "lean",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

    # wait finish
    import time

    for _ in range(60):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            break
    assert st.get("status") == "COMPLETED"

    # verify manifest has profile
    m = client.get(f"/static/jobs/{job_id}/manifest.json")
    assert m.status_code == 200
    manifest = m.json()
    assert manifest.get("profile") == "lean"
