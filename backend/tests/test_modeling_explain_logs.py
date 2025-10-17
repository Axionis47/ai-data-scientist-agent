from fastapi.testclient import TestClient
import io
import pandas as pd
from pathlib import Path
from app.main import app


def test_modeling_explain_and_logs(tmp_path: Path):
    client = TestClient(app)
    df = pd.DataFrame(
        {
            "id": [f"A{i}" for i in range(60)],
            "age": list(range(60)),
            "city": ["X"] * 30 + ["Y"] * 30,
            "y": [0, 1] * 30,
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    r = client.post(
        "/upload", files={"file": ("m.csv", io.BytesIO(csv_bytes), "text/csv")}
    )
    info = r.json()
    job_id = info["job_id"]
    dataset_path = info["dataset_path"]

    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "train a classification model on y",
        "sheet_name": None,
        "delimiter": ",",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

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
    # explain section may include importances or pdp
    assert "explain" in res
