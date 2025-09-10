from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import io
from pathlib import Path
from app.main import app, LARGE_FILE_MB


def test_large_file_chunked_missingness(tmp_path: Path, monkeypatch):
    client = TestClient(app)
    # Construct a DataFrame that would be large if written normally, but we'll monkeypatch _is_large_file
    df = pd.DataFrame({
        'a': np.random.randn(10000),
        'b': np.random.randn(10000),
        'c': [None] * 500 + list(range(9500))
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # Upload
    r = client.post("/upload", files={"file": ("big.csv", io.BytesIO(csv_bytes), "text/csv")})
    info = r.json(); job_id = info['job_id']; dataset_path = info['dataset_path']

    # Force chunked path by monkeypatching _is_large_file to True
    from app import main as m
    monkeypatch.setattr(m, "_is_large_file", lambda p: True)

    # Analyze
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "do eda only",
        "sheet_name": None,
        "delimiter": ","
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

    import time
    for _ in range(30):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st['status'] == 'COMPLETED' or st['stage'] == 'clarify':
            break

    res = client.get(f"/result/{job_id}")
    assert res.status_code == 200
    eda = res.json()['eda']
    assert 'missing' in eda
    assert 'meta' in eda
    # The chunked path sets total_rows_est
    assert 'total_rows_est' in eda['meta']

