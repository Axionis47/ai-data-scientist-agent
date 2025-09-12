from app.core.schemas import validate_report_json


def test_report_json_without_model_card_valid():
    obj = {
        "title": "Report",
        "kpis": {"f1": 0.82},
        "sections": [
            {"heading": "Summary", "items": ["One", "Two"]},
            {"heading": "Details", "html": "<p>ok</p>"},
        ],
    }
    errs = validate_report_json(obj)
    assert errs == []


def test_report_json_with_model_card_valid():
    obj = {
        "title": "Report",
        "kpis": {"r2": 0.71},
        "sections": [{"heading": "Summary", "items": ["A"]}],
        "model_card": {
            "name": "rf_clf",
            "task": "classification",
            "metric_primary": "f1",
            "metric_value": 0.81,
            "features": {"numeric": 5, "categorical": 3},
            "candidates": ["logreg", "rf_clf"],
        },
    }
    errs = validate_report_json(obj)
    assert errs == []


def test_report_json_with_model_card_invalid():
    obj = {
        "title": "Report",
        "kpis": {"f1": 0.5},
        "sections": [{"heading": "S", "items": []}],
        "model_card": {
            # missing name/task/metric fields
        },
    }
    errs = validate_report_json(obj)
    assert any("model_card missing" in e for e in errs)

