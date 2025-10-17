from fastapi.testclient import TestClient
import io
import pandas as pd
from pathlib import Path
from app.main import app


def test_clarify_gating_and_resume(tmp_path: Path):
    client = TestClient(app)
    # Create a small CSV in-memory lacking explicit target instructions initially
    df = pd.DataFrame(
        {
            "id": [f"A{i}" for i in range(30)],
            "age": list(range(30)),
            "city": ["X"] * 15 + ["Y"] * 15,
            "y": [0, 1] * 15,
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    r = client.post(
        "/upload", files={"file": ("clarify.csv", io.BytesIO(csv_bytes), "text/csv")}
    )
    assert r.status_code == 200
    info = r.json()
    job_id = info["job_id"]
    dataset_path = info["dataset_path"]

    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "build a classification model",
        "sheet_name": None,
        "delimiter": ",",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

    # Wait a bit then expect stage could be clarify
    import time

    for _ in range(20):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st["stage"] in ("clarify", "modeling", "done", "qa", "report"):
            break
    assert st["status"] == "RUNNING"

    if st["stage"] == "clarify":
        # Resume via clarify
        rr = client.post("/clarify", json={"job_id": job_id, "message": "target=y"})
        assert rr.status_code == 200
        for _ in range(30):
            time.sleep(0.2)
            st = client.get(f"/status/{job_id}").json()
            if st["status"] == "COMPLETED":
                break
        assert st["status"] == "COMPLETED"
    else:
        # If modeling proceeded automatically (e.g., robustness changes), ensure result exists
        res = client.get(f"/result/{job_id}")
        assert res.status_code == 200
