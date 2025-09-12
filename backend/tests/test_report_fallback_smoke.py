from app.reporting.report import _fallback_report_html

def test_fallback_report_smoke_contains_sections():
    html = _fallback_report_html("jobabc", {}, {"task": "classification", "best": {"f1": 0.5}}, {"pdp": []})
    assert "Analysis Report" in html
    for anchor in ("id='explainability'", "id='risks'", "id='data-caveats'"):
        assert anchor in html

