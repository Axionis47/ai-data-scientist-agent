from fastapi.testclient import TestClient
from pathlib import Path
from app.main import app


def test_analyze_rejects_outside_jobs_dir(tmp_path: Path):
    client = TestClient(app)
    # Create a stray file outside JOBS_DIR
    stray = tmp_path / "outside.csv"
    stray.write_text("a,b\n1,2\n")

    # Try to analyze using that path
    body = {
        "dataset_path": str(stray),
        "file_format": "csv",
        "nl_description": "demo",
        "question": "model it",
        "sheet_name": None,
        "delimiter": ",",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 400
