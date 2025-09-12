from fastapi.testclient import TestClient
from app.main import app

def test_modeling_fallback_when_sklearn_missing(monkeypatch):
    client = TestClient(app)
    # Upload a tiny CSV
    import io
    import pandas as pd
    csv_bytes = pd.DataFrame({"x":[1,2,3],"y":[0,1,0]}).to_csv(index=False).encode("utf-8")
    r = client.post("/upload", files={"file": ("t.csv", io.BytesIO(csv_bytes), "text/csv")})
    info = r.json(); job_id = info['job_id']; dataset_path = info['dataset_path']

    # Force run_modeling import to fail
    import app.pipeline.run as pr
    def bad_import(*args, **kwargs):
        raise ImportError("sklearn not installed")
    monkeypatch.setattr(pr, "reporting_expert", lambda *a, **k: "<html></html>")
    monkeypatch.setattr(pr, "__import__", bad_import, raising=False)

    # Analyze (should complete with modeling error but not crash)
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "do eda only",
        "sheet_name": None,
        "delimiter": ",",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

