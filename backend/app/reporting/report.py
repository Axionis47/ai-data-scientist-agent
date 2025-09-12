import json

"""Reporting engine: builds report_html via JSON-first LLM path or deterministic fallback.

Key function: reporting_expert(job_id, eda, modeling, explain) -> HTML
- JSON-first: requests structured JSON, validates shape, renders deterministic HTML
- HTML via LLM: prompts for clean inline-CSS HTML when OpenAI is configured
- Fallback: no-LLM deterministic HTML; now includes model card, risks/caveats, and anchor sections
- Extras: _inline_img for optional base64 inlining; _span for OTEL spans
"""

import os
from typing import Dict, Any
from ..core.logs import model_decision
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

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency for CI
    OpenAI = None  # type: ignore

from ..core.config import REPORT_INLINE_ASSETS
import base64, pathlib
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
    title = j.get("title") or f"Job {job_id} Report"
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


def reporting_expert(
    job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], explain: Dict[str, Any]
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    # JSON-first path (feature-flagged)
    if REPORT_JSON_FIRST and api_key and OpenAI is not None:
        with _span("report.json_first", {"job_id": job_id, "enabled": True}):
            try:
                client = OpenAI()
                jprompt = {
                    "role": "user",
                    "content": (
                        "Return ONLY JSON with keys: title (str), kpis (object of numeric values), "
                        "sections (array of objects each with heading and either items[] or html string). "
                        "No markdown, no prose outside JSON. Context:"
                        + json.dumps(
                            {
                                "job_id": job_id,
                                "kpis": modeling.get("best") or {},
                                "task": modeling.get("task"),
                                "notes": (modeling.get("notes") or []),
                                "explain": {
                                    "roc": explain.get("roc"),
                                    "pr": explain.get("pr"),
                                    "pdp": explain.get("pdp"),
                                },
                            }
                        )[:4000]
                    ),
                }
                rj = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You strictly output JSON. No extra text.",
                        },
                        jprompt,
                    ],
                    temperature=0.1,
                )
                raw = rj.choices[0].message.content or "{}"
                # Best-effort strip code fences
                raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
                obj = json.loads(raw)
                errs = validate_report_json(obj)
                if not errs:
                    model_decision(job_id, "Reporting: JSON-first report validated")
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

    # HTML path (OpenAI or fallback)
    if not api_key or OpenAI is None:
        with _span("report.fallback", {"job_id": job_id, "reason": "no_openai"}):
            model_decision(
                job_id,
                "Reporting: fallback template used (OpenAI unavailable or no OPENAI_API_KEY)",
            )
            return _fallback_report_html(job_id, eda, modeling, explain)
    try:
        with _span("report.html_llm", {"job_id": job_id}):
            client = OpenAI()
            ctx = {
                "job_id": job_id,
                "kpis": modeling.get("best") or {},
                "task": modeling.get("task"),
                "notes": (modeling.get("notes") or []),
                "explain": {
                    "roc": explain.get("roc"),
                    "pr": explain.get("pr"),
                    "pdp": explain.get("pdp"),
                },
                "brand": {
                    "primary": REPORT_PRIMARY,
                    "accent": REPORT_ACCENT,
                    "bg": REPORT_BG,
                    "surface": REPORT_SURFACE,
                    "text": REPORT_TEXT,
                    "muted": REPORT_MUTED,
                    "font": REPORT_FONT_FAMILY,
                    "logo": REPORT_LOGO_URL,
                },
            }
            prompt = {
                "role": "user",
                "content": f"Generate clean, self-contained HTML with inline CSS for an executive summary. Use brand tokens if provided. Include headings, KPIs, bullets, and embed ROC/PR images if present. Context: {json.dumps(ctx)[:4000]}",
            }
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise, reliable report writer. Return HTML only.",
                    },
                    prompt,
                ],
                temperature=0.2,
            )
            html = r.choices[0].message.content or ""
            if "<html" not in html and "<div" not in html:
                raise ValueError("router returned non-HTML")
            model_decision(job_id, "Reporting: generated via OpenAI")
            return html
    except Exception as e:
        with _span(
            "report.fallback_on_error", {"job_id": job_id, "error": str(e)[:200]}
        ):
            model_decision(job_id, f"Reporting failed: {e}; using fallback")
            return _fallback_report_html(job_id, eda, modeling, explain)
