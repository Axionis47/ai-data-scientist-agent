import pandas as pd

from app.services.feature_engineering_service import add_datetime_features


def test_add_datetime_features_adds_columns():
    base = pd.to_datetime("2024-01-01")
    dates = pd.date_range(base, periods=5, freq="D")
    df = pd.DataFrame({"ds": dates, "x": [1, 2, 3, 4, 5]})
    eda = {"time_columns": ["ds"]}
    out, rep = add_datetime_features(df, eda, {})
    assert "ds_year" in out.columns
    assert "ds_month" in out.columns
    assert "ds_dow" in out.columns
    assert set(rep["added"]) == {"ds_year", "ds_month", "ds_dow"}
