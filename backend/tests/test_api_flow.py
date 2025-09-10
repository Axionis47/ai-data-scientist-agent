import json
from pathlib import Path
import io

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, JOBS_DIR


def test_api_upload_analyze_clarify_and_result(tmp_path: Path):
    client = TestClient(app)
    # Create a small CSV in-memory
    df = pd.DataFrame({
        'id': [f'A{i}' for i in range(20)],
        'age': list(range(20)),
        'city': ['X']*10 + ['Y']*10,
        'y': [0,1]*10
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # Upload
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    info = r.json()
    job_id = info['job_id']
    dataset_path = info['dataset_path']

    # Analyze without target to trigger clarify
    body = {
        "job_id": job_id,
        "dataset_path": dataset_path,
        "file_format": "csv",
        "nl_description": "demo",
        "question": "please build a classification model",
        "sheet_name": None,
        "delimiter": ","
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200

    # Wait briefly for EDA to complete and clarify to be set
    import time
    time.sleep(0.5)
    r = client.get(f"/status/{job_id}")
    assert r.status_code == 200
    st = r.json()
    assert st['status'] == 'RUNNING'
    # Clarify with target
    r = client.post("/clarify", json={"job_id": job_id, "message": "target=y"})
    assert r.status_code == 200

    # Wait for pipeline to resume and finish
    for _ in range(30):
        time.sleep(0.2)
        st = client.get(f"/status/{job_id}").json()
        if st['status'] == 'COMPLETED':
            break
    assert st['status'] == 'COMPLETED'

    # Check result and static plots
    r = client.get(f"/result/{job_id}")
    assert r.status_code == 200
    res = r.json()
    assert 'eda' in res
    # Modeling section should exist even if best is None
    assert 'modeling' in res
    # If best exists, it has a name
    if res['modeling'].get('best'):
        assert 'name' in res['modeling']['best']
    plots = ((res['eda'] or {}).get('plots') or {})
    if 'missingness' in plots:
        img = client.get(plots['missingness'])
        assert img.status_code == 200

