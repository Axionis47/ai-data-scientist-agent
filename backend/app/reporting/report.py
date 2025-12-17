import json

"""Reporting engine: builds report_html via JSON-first LLM path or deterministic fallback.

Key function: reporting_expert(job_id, eda, modeling, explain) -> HTML
- JSON-first: requests structured JSON, validates shape, renders deterministic HTML
- HTML via LLM: prompts for clean inline-CSS HTML when LLM provider is configured
- Fallback: no-LLM deterministic HTML; now includes model card, risks/caveats, and anchor sections
- Extras: _inline_img for optional base64 inlining; _span for OTEL spans
"""

import os
from typing import Dict, Any
from ..core.logs import model_decision
from ..core.llm import get_llm_client
from ..core.config import (
    REPORT_PRIMARY,
    REPORT_ACCENT,
    REPORT_BG,
    REPORT_SURFACE,
    REPORT_TEXT,
    REPORT_MUTED,
    REPORT_OK,
    REPORT_WARN,
    REPORT_ERROR,
    REPORT_FONT_FAMILY,
    REPORT_LOGO_URL,
)

import base64
import pathlib
from contextlib import contextmanager

try:
    from opentelemetry import trace  # type: ignore
except Exception:
    trace = None  # type: ignore


@contextmanager
def _span(name: str, attributes: Dict[str, Any] | None = None):
    if trace is not None:
        tracer = trace.get_tracer("app.reporting")
        with tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    try:
                        s.set_attribute(k, v)
                    except Exception:
                        pass
            yield
    else:
        yield


def _inline_img(path_or_url: str) -> str:
    try:
        # Read flag dynamically to support test monkeypatch of env
        inline_enabled = os.getenv("REPORT_INLINE_ASSETS", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if not inline_enabled:
            return path_or_url
        if not path_or_url or path_or_url.startswith("http"):
            return path_or_url
        # Paths are expected like /static/jobs/<id>/plots/foo.png
        # Map to filesystem under JOBS_DIR
        p = path_or_url
        if p.startswith("/static/jobs/"):
            # /static/jobs/<id>/plots/x.png -> backend/app/data/jobs/<id>/plots/x.png via config JOBS_DIR
            from ..core.config import JOBS_DIR

            parts = p.split("/static/jobs/")[-1].split("/")
            # parts like [<id>, plots, x.png]
            fs = JOBS_DIR / parts[0] / "/".join(parts[1:])
        else:
            fs = pathlib.Path(p)
        data = pathlib.Path(fs).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        ext = pathlib.Path(fs).suffix.lstrip(".") or "png"
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return path_or_url


def _render_model_card(job_id: str, modeling: Dict[str, Any]) -> str:
    best = modeling.get("best") or {}
    task = modeling.get("task")
    feats = modeling.get("features") or {}
    cands = modeling.get("selected_tools") or []
    metric_primary = (
        "f1"
        if task == "classification"
        else ("r2" if (best.get("r2") is not None) else "rmse")
    )
    metric_value = best.get(metric_primary)
    rows = [
        ("Model", best.get("name") or "N/A"),
        ("Task", task or "N/A"),
        (
            "Primary",
            f"{metric_primary}: {metric_value}" if metric_value is not None else "N/A",
        ),
        (
            "Features",
            f"{feats.get('numeric',0)} numeric, {feats.get('categorical',0)} categorical",
        ),
        ("Candidates", ", ".join(cands) if cands else "N/A"),
    ]
    cells = "".join(
        [
            f"<div><div style='font-weight:600'>{k}</div><div>{v}</div></div>"
            for k, v in rows
        ]
    )
    return f"<div class='card'><div class='h1'>Model Card</div><div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px'>{cells}</div></div>"


from ..core.config import REPORT_JSON_FIRST
from ..core.schemas import validate_report_json


def _render_from_json(job_id: str, j: dict) -> str:
    # Minimal deterministic HTML from a validated JSON report object
    j.get("title") or f"Job {job_id} Report"
    kpi_html = "".join(
        [
            f"<div class='card'><div class='h1'>{k}</div><div class='kpi'>{v}</div></div>"
            for k, v in (j.get("kpis") or {}).items()
        ]
    )
    sections = []
    for s in j.get("sections") or []:
        if "html" in s:
            sections.append(
                f"<div class='card section'><div class='h1'>{s.get('heading') or ''}</div>{s.get('html')}</div>"
            )
        else:
            items = "".join([f"<li>{str(it)}</li>" for it in (s.get("items") or [])])
            sections.append(
                f"<div class='card section'><div class='h1'>{s.get('heading') or ''}</div><ul>{items}</ul></div>"
            )
    # Optional model card in JSON
    mc = j.get("model_card")
    model_card_html = ""
    if isinstance(mc, dict):
        rows = []
        rows.append(("Model", mc.get("name", "N/A")))
        rows.append(("Task", mc.get("task", "N/A")))
        mp = mc.get("metric_primary")
        mv = mc.get("metric_value")
        rows.append(("Primary", f"{mp}: {mv}" if (mp and mv is not None) else "N/A"))
        feats = mc.get("features") or {}
        rows.append(
            (
                "Features",
                f"{feats.get('numeric',0)} numeric, {feats.get('categorical',0)} categorical",
            )
        )
        cands = mc.get("candidates") or []
        rows.append(("Candidates", ", ".join(cands) if cands else "N/A"))
        if mc.get("threshold") is not None:
            rows.append(("Threshold", mc.get("threshold")))
        cells = "".join(
            [
                f"<div><div style='font-weight:600'>{k}</div><div>{v}</div></div>"
                for k, v in rows
            ]
        )
        model_card_html = f"<div class='card section'><div class='h1'>Model Card</div><div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px'>{cells}</div></div>"
    base = _fallback_report_html(job_id, {}, {}, {})
    base = base.replace(
        "<div class='grid section'>", f"<div class='grid section'>{kpi_html}"
    )
    return base + "".join(sections) + model_card_html


def _fallback_report_html(
    job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any]
) -> str:
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
    kpi = (
        "F1: {:.3f}".format(best.get("f1", 0.0))
        if task == "classification"
        else "RMSE: {:.3f}".format(best.get("rmse", 0.0))
    )
    pdps = "".join(
        [
            f'<div class="card"><img src="{_inline_img(p)}"/></div>'
            for p in (explain.get("pdp") or [])
        ]
    )
    roc_html = (
        f'<div class="card"><img src="{_inline_img(explain.get("roc"))}"/></div>'
        if explain.get("roc")
        else ""
    )
    pr_html = (
        f'<div class="card"><img src="{_inline_img(explain.get("pr"))}"/></div>'
        if explain.get("pr")
        else ""
    )
    brand = (
        f"<div class='brand'><img src='{REPORT_LOGO_URL}'/><div class='muted'>Auto‑generated analysis</div></div>"
        if REPORT_LOGO_URL
        else ""
    )
    card_html = _render_model_card(job_id, modeling)
    return f"""
    {css}
    <div class='card'>
      {brand}
      <div class='h1'>Analysis Report <span class='tag'>Job {job_id[:8]}</span></div>
      <div class='muted'>Task: {task}; Best: {best.get('name','n/a')}; {('Tuned threshold: '+str(best.get('tuned_threshold'))) if best.get('tuned_threshold') is not None else ''}</div>
    </div>
    {card_html}
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
    <div class='card section' id='explainability'>
      <div class='h1'>Explainability</div>
      <pre style='white-space:pre-wrap' class='muted'>{json.dumps(explain.get('importances_top') or explain.get('importances') or {}, indent=2)}</pre>
      <div class='grid'>{roc_html}{pr_html}{pdps}</div>
    </div>
    <div class='card section' id='risks'>
      <div class='h1'>Risks & Limitations</div>
      <ul>
        <li>Automated modeling on small samples may be unstable; validate on full data.</li>
        <li>Feature leakage checks are limited; audit target leakage manually.</li>
        <li>Class imbalance can affect metrics; consider threshold tuning and stratified splits.</li>
      </ul>
    </div>
    <div class='card section' id='data-caveats'>
      <div class='h1'>Data Caveats</div>
      <ul>
        <li>Missing values are imputed; verify imputation strategy is appropriate.</li>
        <li>Categorical high-cardinality is reduced via Top-K; rare categories grouped as __OTHER__.</li>
        <li>Encoding and delimiter are auto-detected; confirm parsing matched expectations.</li>
      </ul>
    </div>
    """


def _build_enriched_report_context(
    job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any], manifest: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Build enriched context for LLM report generation with comprehensive insights."""
    manifest = manifest or {}
    best = modeling.get("best") or {}
    task = modeling.get("task")

    # Dataset summary
    shape = eda.get("shape") or {}
    missing = eda.get("missing") or {}
    high_missing = [(k, round(v.get("pct", 0), 1)) for k, v in missing.items() if v.get("pct", 0) > 5]

    # Column type breakdown
    dtypes = eda.get("dtypes") or {}
    numeric_cols = [k for k, v in dtypes.items() if "float" in v or "int" in v]
    categorical_cols = [k for k, v in dtypes.items() if "object" in v or "category" in v]

    # Feature importances (top 10) with formatting
    importances = explain.get("importances") or []
    if isinstance(importances, list):
        top_features = [
            {"feature": f.get("feature") or f.get("name"), "importance": round(f.get("importance") or f.get("score", 0), 4)}
            for f in importances[:10]
            if isinstance(f, dict)
        ]
    else:
        top_features = []

    # Leaderboard comparison with scores
    leaderboard = modeling.get("leaderboard") or []
    model_comparison = [
        {
            "rank": i + 1,
            "name": m.get("name"),
            "f1": round(m.get("f1", 0), 3) if m.get("f1") else None,
            "accuracy": round(m.get("acc", 0), 3) if m.get("acc") else None,
            "cv_mean": round(m.get("cv_mean", 0), 3) if m.get("cv_mean") else None,
        }
        for i, m in enumerate(leaderboard[:5])
    ]

    # Confusion matrix summary (if available)
    confusion_matrix = None
    cm = explain.get("confusion_matrix") or modeling.get("confusion_matrix")
    if cm and isinstance(cm, (list, dict)):
        confusion_matrix = cm

    # Class distribution (for classification)
    class_distribution = None
    target_dist = eda.get("target_distribution")
    if target_dist:
        class_distribution = target_dist
    elif task == "classification":
        # Try to infer from modeling notes
        notes = modeling.get("notes") or []
        for note in notes:
            if "imbalance" in str(note).lower():
                class_distribution = {"note": note}
                break

    # Top correlations
    top_correlations = eda.get("top_correlations") or []
    correlation_insights = [
        {"pair": f"{c[0]} ↔ {c[1]}", "correlation": round(c[2], 3)}
        for c in top_correlations[:5]
        if len(c) >= 3
    ]

    # Data quality issues and recommendations
    recommendations = eda.get("recommendations") or []
    data_quality = manifest.get("data_quality") or {}
    dq_issues = data_quality.get("issues") or []
    quality_summary = {
        "recommendations": recommendations[:5],
        "issues": [{"id": i.get("id"), "severity": i.get("severity")} for i in dq_issues[:3]],
    }

    # Outlier summary
    outlier_info = None
    skew = eda.get("skew") or {}
    if skew:
        high_skew = [(k, round(v, 2)) for k, v in skew.items() if abs(v) > 2]
        if high_skew:
            outlier_info = {"high_skew_features": high_skew[:5], "note": "Features with |skew|>2 may have outliers"}

    # Feature engineering summary
    fe_info = manifest.get("feature_engineering") or {}
    fe_summary = None
    if fe_info:
        added_features = []
        for key in ["datetime", "timeseries", "text"]:
            sub = fe_info.get(key) or {}
            added = sub.get("added") or []
            if added:
                added_features.extend(added[:3])
        if added_features:
            fe_summary = {"engineered_features": added_features[:5]}

    # Model training notes
    model_notes = modeling.get("notes") or []

    return {
        "job_id": job_id,
        "dataset": {
            "rows": shape.get("rows"),
            "cols": shape.get("cols"),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "high_missing_cols": high_missing[:5],
        },
        "model": {
            "name": best.get("name"),
            "task": task,
            "metrics": {
                "f1": round(best.get("f1"), 3) if best.get("f1") else None,
                "accuracy": round(best.get("acc"), 3) if best.get("acc") else None,
                "roc_auc": round(best.get("roc_auc"), 3) if best.get("roc_auc") else None,
                "pr_auc": round(best.get("pr_auc"), 3) if best.get("pr_auc") else None,
                "r2": round(best.get("r2"), 3) if best.get("r2") else None,
                "rmse": round(best.get("rmse"), 3) if best.get("rmse") else None,
            },
            "threshold": round(best.get("thr"), 3) if best.get("thr") else None,
            "cv_mean": round(best.get("cv_mean"), 3) if best.get("cv_mean") else None,
            "cv_std": round(best.get("cv_std"), 3) if best.get("cv_std") else None,
        },
        "top_features": top_features,
        "model_comparison": model_comparison,
        "confusion_matrix": confusion_matrix,
        "class_distribution": class_distribution,
        "correlation_insights": correlation_insights,
        "data_quality": quality_summary,
        "outlier_info": outlier_info,
        "feature_engineering": fe_summary,
        "model_notes": model_notes[:5],
        "user_question": manifest.get("question"),
        "business_context": manifest.get("nl_description") or manifest.get("context"),
        "explain": {
            "roc": explain.get("roc"),
            "pr": explain.get("pr"),
            "pdp": explain.get("pdp")[:3] if explain.get("pdp") else None,  # Limit PDPs
            "shap": explain.get("shap"),
        },
    }


def reporting_expert(
    job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any], manifest: Dict[str, Any] = None
) -> str:
    llm_client = get_llm_client()
    manifest = manifest or {}

    # JSON-first path (feature-flagged)
    if REPORT_JSON_FIRST and llm_client is not None:
        with _span("report.json_first", {"job_id": job_id, "enabled": True}):
            try:
                enriched_ctx = _build_enriched_report_context(job_id, eda, modeling, explain, manifest)

                system_prompt = """You are an expert data science report writer creating executive summaries for business stakeholders.

Output ONLY valid JSON with this structure:
{
  "title": "Clear, descriptive title",
  "kpis": {"metric_name": numeric_value, ...},
  "sections": [
    {"heading": "Section Title", "items": ["bullet point 1", "bullet point 2"]},
    {"heading": "Another Section", "html": "<p>HTML content</p>"}
  ]
}

Guidelines:
- Write for business stakeholders, not data scientists
- Explain what metrics mean in business terms
- Highlight actionable insights
- Note any data quality concerns that affect reliability
- Keep bullet points concise but informative"""

                user_prompt = f"""Generate an executive summary report for this ML analysis.

## Required Sections:
1. **Model Performance Summary** - Key metrics with business interpretation
2. **Top Predictive Factors** - What drives the predictions (from top_features)
3. **Model Comparison** - How different models performed (from model_comparison)
4. **Data Quality Notes** - Any issues that affect reliability
5. **Business Recommendations** - Actionable next steps based on findings

## Context Data:
{json.dumps(enriched_ctx, indent=2)[:7000]}

Generate the JSON report now."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                raw = llm_client.chat(messages=messages, temperature=0.1, json_mode=True)
                # Best-effort strip code fences (shouldn't be needed with json_mode)
                raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
                obj = json.loads(raw)
                errs = validate_report_json(obj)
                if not errs:
                    model_decision(job_id, f"Reporting: JSON-first report validated ({llm_client.provider_name})")
                    return _render_from_json(job_id, obj)
                else:
                    model_decision(
                        job_id,
                        f"Reporting: JSON-first invalid, errs={errs}; falling back to HTML mode",
                    )
            except Exception as e:
                model_decision(
                    job_id, f"Reporting: JSON-first failed: {e}; falling back"
                )

    # HTML path (LLM or fallback)
    if llm_client is None:
        with _span("report.fallback", {"job_id": job_id, "reason": "no_llm"}):
            model_decision(
                job_id,
                "Reporting: fallback template used (no LLM provider available)",
            )
            return _fallback_report_html(job_id, eda, modeling, explain)
    try:
        with _span("report.html_llm", {"job_id": job_id}):
            # Use enriched context for better reports
            enriched_ctx = _build_enriched_report_context(job_id, eda, modeling, explain, manifest)
            enriched_ctx["brand"] = {
                "primary": REPORT_PRIMARY,
                "accent": REPORT_ACCENT,
                "bg": REPORT_BG,
                "surface": REPORT_SURFACE,
                "text": REPORT_TEXT,
                "muted": REPORT_MUTED,
                "font": REPORT_FONT_FAMILY,
                "logo": REPORT_LOGO_URL,
            }

            system_prompt = """You are an expert data science report writer creating executive summaries.

Return ONLY clean, self-contained HTML with inline CSS. No markdown, no code blocks.

Structure your report with these sections:
1. Header with title and key metrics
2. Model Performance (with interpretation)
3. Top Predictive Factors (from feature importances)
4. Model Comparison table (if multiple models)
5. Data Quality Notes (any concerns)
6. Business Recommendations (actionable insights)

Use the brand tokens for colors and fonts. Keep it professional and concise."""

            user_prompt = f"""Generate an executive summary HTML report.

## Brand Tokens:
- Primary color: {REPORT_PRIMARY}
- Accent color: {REPORT_ACCENT}
- Background: {REPORT_BG}
- Text: {REPORT_TEXT}
- Font: {REPORT_FONT_FAMILY}

## Analysis Results:
{json.dumps(enriched_ctx, indent=2)[:7000]}

Generate the HTML report now. Start with <div> or <html>."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            html = llm_client.chat(messages=messages, temperature=0.2)
            if "<html" not in html and "<div" not in html:
                raise ValueError("LLM returned non-HTML")
            model_decision(job_id, f"Reporting: generated via {llm_client.provider_name}")
            return html
    except Exception as e:
        with _span(
            "report.fallback_on_error", {"job_id": job_id, "error": str(e)[:200]}
        ):
            model_decision(job_id, f"Reporting failed: {e}; using fallback")
            return _fallback_report_html(job_id, eda, modeling, explain)
