import json
import os
from typing import Dict, Any
from ..core.logs import model_decision
from ..core.config import (
    REPORT_PRIMARY, REPORT_ACCENT, REPORT_BG, REPORT_SURFACE, REPORT_TEXT,
    REPORT_MUTED, REPORT_OK, REPORT_WARN, REPORT_ERROR, REPORT_FONT_FAMILY,
    REPORT_LOGO_URL
)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency for CI
    OpenAI = None  # type: ignore


def _fallback_report_html(job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any]) -> str:
    css = f"""
    <style>
      :root{{ --bg:{REPORT_BG or '#0b0f14'}; --surface:{REPORT_SURFACE or '#10161b'}; --text:{REPORT_TEXT or '#e6edf3'}; --muted:{REPORT_MUTED or '#94a3b8'}; --primary:{REPORT_PRIMARY or '#3b82f6'}; --accent:{REPORT_ACCENT or '#22d3ee'}; --ok:{REPORT_OK or '#22c55e'}; --warn:{REPORT_WARN or '#f59e0b'}; --error:{REPORT_ERROR or '#ef4444'}; }}
      body{{font-family:{REPORT_FONT_FAMILY or '-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Inter,Helvetica Neue,Arial,sans-serif'};background:var(--bg);color:var(--text);margin:0;padding:24px;}}
      .h1{{font-size:24px;font-weight:700;margin:0 0 8px}}
      .muted{{color:var(--muted)}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
      .card{{background:var(--surface);border:1px solid rgba(148,163,184,.2);border-radius:10px;padding:16px}}
      .kpi{{font-size:28px;font-weight:700}}
      .tag{{display:inline-block;background:rgba(59,130,246,.15);color:var(--primary);border:1px solid rgba(59,130,246,.35);padding:2px 8px;border-radius:999px;font-size:12px;margin-left:8px}}
      img{{max-width:100%;border-radius:6px;border:1px solid rgba(148,163,184,.2)}}
      .section{{margin-top:16px}}
      .brand{{display:flex;align-items:center;gap:8px;margin-bottom:12px}}
      .brand img{{height:28px}}
    </style>
    """
    best = (modeling or {}).get("best") or {}
    task = (modeling or {}).get("task") or "descriptive"
    kpi = "F1: {:.3f}".format(best.get("f1", 0.0)) if task=="classification" else "RMSE: {:.3f}".format(best.get("rmse", 0.0))
    pdps = "".join([f'<div class="card"><img src="/static/jobs/{job_id}/plots/{p.split("/")[-1]}"/></div>' for p in (explain.get("pdp") or [])])
    roc = explain.get("roc"); pr = explain.get("pr")
    brand = f"<div class='brand'><img src='{REPORT_LOGO_URL}'/><div class='muted'>Auto‑generated analysis</div></div>" if REPORT_LOGO_URL else ""
    return f"""
    {css}
    <div class='card'>
      {brand}
      <div class='h1'>Analysis Report <span class='tag'>Job {job_id[:8]}</span></div>
      <div class='muted'>Task: {task}; Best: {best.get('name','n/a')}; {('Tuned threshold: '+str(best.get('tuned_threshold'))) if best.get('tuned_threshold') is not None else ''}</div>
    </div>
    <div class='grid section'>
      <div class='card'>
        <div class='h1'>Key Metric</div>
        <div class='kpi'>{kpi}</div>
      </div>
      <div class='card'>
        <div class='h1'>Modeling Summary</div>
        <pre style='white-space:pre-wrap'>{json.dumps(modeling, indent=2)}</pre>
      </div>
    </div>
    <div class='card section'>
      <div class='h1'>Explainability</div>
      <pre style='white-space:pre-wrap' class='muted'>{json.dumps(explain.get('importances_top') or explain.get('importances') or {}, indent=2)}</pre>
      <div class='grid'>{pdps}</div>
    </div>
    """


def reporting_expert(job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        model_decision(job_id, "Reporting: fallback template used (OpenAI unavailable or no OPENAI_API_KEY)")
        return _fallback_report_html(job_id, eda, modeling, explain)
    try:
        client = OpenAI()
        ctx = {
            "job_id": job_id,
            "kpis": modeling.get("best") or {},
            "task": modeling.get("task"),
            "notes": (modeling.get("notes") or []),
            "explain": {"roc": explain.get("roc"), "pr": explain.get("pr"), "pdp": explain.get("pdp")},
            "brand": {
                "primary": REPORT_PRIMARY,
                "accent": REPORT_ACCENT,
                "bg": REPORT_BG,
                "surface": REPORT_SURFACE,
                "text": REPORT_TEXT,
                "muted": REPORT_MUTED,
                "font": REPORT_FONT_FAMILY,
                "logo": REPORT_LOGO_URL,
            }
        }
        prompt = {
            "role": "user",
            "content": f"Generate clean, self-contained HTML with inline CSS for an executive summary. Use brand tokens if provided. Include headings, KPIs, bullets, and embed ROC/PR images if present. Context: {json.dumps(ctx)[:4000]}"
        }
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a precise, reliable report writer. Return HTML only."}, prompt],
            temperature=0.2
        )
        html = r.choices[0].message.content or ""
        if "<html" not in html and "<div" not in html:
            raise ValueError("router returned non-HTML")
        model_decision(job_id, "Reporting: generated via OpenAI")
        return html
    except Exception as e:
        model_decision(job_id, f"Reporting failed: {e}; using fallback")
        return _fallback_report_html(job_id, eda, modeling, explain)

