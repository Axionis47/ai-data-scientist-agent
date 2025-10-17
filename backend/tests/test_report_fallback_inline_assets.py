from pathlib import Path
from app.reporting.report import _fallback_report_html


def test_fallback_report_includes_sections_and_inline_when_flag(tmp_path, monkeypatch):
    # Enable inlining
    monkeypatch.setenv("REPORT_INLINE_ASSETS", "true")
    # Create fake job dir and plot
    job_id = "jobxyz"
    root = Path(__file__).resolve().parents[2]
    jobs_dir = root / "backend" / "data" / "jobs" / job_id / "plots"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    img = jobs_dir / "pdp_age.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny PNG header
    explain = {"pdp": [f"/static/jobs/{job_id}/plots/pdp_age.png"]}
    html = _fallback_report_html(
        job_id, {}, {"task": "classification", "best": {"f1": 0.7}}, explain
    )
    assert "id='explainability'" in html
    assert "id='risks'" in html and "id='data-caveats'" in html
    assert "data:image/" in html
