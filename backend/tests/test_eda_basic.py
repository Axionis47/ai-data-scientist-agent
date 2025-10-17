from pathlib import Path
import pandas as pd

from app.main import compute_eda, compute_target_relations, compute_timeseries_hints


def test_compute_eda_basic(tmp_path: Path):
    df = pd.DataFrame(
        {
            "id": [f"A{i}" for i in range(50)],
            "age": [20] * 45 + [None] * 5,
            "city": ["X"] * 40 + ["Y"] * 10,
            "amt": list(range(50)),
        }
    )
    eda = compute_eda(df)
    assert eda["shape"]["rows"] == 50
    assert "id" in eda["id_candidates"]
    assert eda["missing"]["age"]["count"] == 5
    assert "numeric_stats" in eda
    assert "categorical_topk" in eda


def test_target_relations(tmp_path: Path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "city": ["A", "A", "B", "B", "C", "C", "C", "A"],
            "y": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )
    rel = compute_target_relations(df, "y")
    assert rel["task"] == "classification"
    assert len(rel["numeric"]) >= 1
    assert len(rel["categorical"]) >= 1


def test_timeseries_hints(tmp_path: Path):
    import pandas as pd

    base = pd.to_datetime("2024-01-01")
    dates = pd.date_range(base, periods=40, freq="D")
    df = pd.DataFrame({"ds": dates, "y": list(range(40))})
    hints = compute_timeseries_hints(df, "ds", "y")
    assert hints["time_col"] == "ds"
    assert hints["points"] >= 30
