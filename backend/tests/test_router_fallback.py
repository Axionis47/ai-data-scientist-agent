from fastapi.testclient import TestClient
import io
import pandas as pd
from pathlib import Path
from app.main import app


def test_router_fallback_without_key(tmp_path: Path, monkeypatch):
    client = TestClient(app)
    # Ensure key absent
    monkeypatch.setenv("OPENAI_API_KEY", "")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8], "y": [0, 1, 0, 1, 0, 1, 0, 1]})
    csv = df.to_csv(index=False).encode("utf-8")
    r = client.post("/upload", files={"file": ("r.csv", io.BytesIO(csv), "text/csv")})
    info = r.json()
    job_id = info["job_id"]
    dataset_path = info["dataset_path"]
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "ctx",
        "question": "classify y",
        "delimiter": ",",
    }
    client.post("/analyze", json=body)
    import time

    for _ in range(60):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st["status"] in ("COMPLETED", "FAILED"):
            break
    res = client.get(f"/result/{job_id}")
    assert res.status_code == 200
    res = res.json()
    assert "modeling" in res
    assert "explain" in res
