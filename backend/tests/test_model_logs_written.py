from fastapi.testclient import TestClient
import io
import pandas as pd
from pathlib import Path
from app.main import app


def test_model_decisions_log(tmp_path: Path):
    client = TestClient(app)
    df = pd.DataFrame(
        {"a": list(range(40)), "b": ["X"] * 20 + ["Y"] * 20, "y": [0, 1] * 20}
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    r = client.post(
        "/upload", files={"file": ("log.csv", io.BytesIO(csv_bytes), "text/csv")}
    )
    job_id = r.json()["job_id"]
    dataset_path = r.json()["dataset_path"]
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "model",
        "delimiter": ",",
    }
    client.post("/analyze", json=body)
    import time

    for _ in range(30):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st["status"] == "COMPLETED":
            break
    # Check model_decisions.log exists and has content
    from pathlib import Path as P

    p = P("backend/data/jobs") / job_id / "logs" / "model_decisions.log"
    assert p.exists()
    content = p.read_text().strip()
    assert content
