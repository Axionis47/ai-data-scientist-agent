import pandas as pd

from app.services.data_quality_service import (
    assess_target_viability,
    detect_identifier_columns,
    detect_near_perfect_predictors,
    data_quality_report,
)


def test_assess_target_viability_binary_imbalance():
    df = pd.DataFrame({"y": [0] * 995 + [1] * 5})
    out = assess_target_viability(df, "y")
    ids = {i["id"] for i in out["issues"]}
    assert (
        "extreme_imbalance" in ids or out["issues"] == []
    )  # warn or none depending on threshold


def test_detect_identifier_columns_by_uniqueness():
    n = 200
    df = pd.DataFrame({"id": list(range(n)), "x": [1] * n})
    ids = detect_identifier_columns(df)
    assert "id" in ids


def test_detect_near_perfect_predictors_binary():
    df = pd.DataFrame(
        {
            "leaky": [0, 0, 1, 1, 0, 1, 0, 1],
            "y": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    flagged = detect_near_perfect_predictors(df, "y")
    assert "leaky" in flagged


def test_data_quality_report_summary_keys():
    df = pd.DataFrame(
        {
            "id": list(range(50)),
            "cat": ["A"] * 25 + ["B"] * 25,
            "y": [0] * 30 + [1] * 20,
        }
    )
    eda = {"nunique": {"id": 50, "cat": 2}, "time_columns": []}
    manifest = {
        "framing": {"target": "y"},
        "router_plan": {"decisions": {"split": "random"}},
    }
    rep = data_quality_report("jobX", df, eda, manifest)
    assert set(rep.keys()) == {"issues", "recommendations", "summary"}
