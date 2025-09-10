import json
import os
from typing import Dict, Any
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency for CI
    OpenAI = None  # type: ignore
from ..core.logs import model_decision


def build_context_pack(eda: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "shape": eda.get("shape"),
        "dtypes": eda.get("dtypes"),
        "missing_summary": {k: v.get("pct") for k, v in (eda.get("missing") or {}).items()},
        "nunique": eda.get("nunique"),
        "id_candidates": eda.get("id_candidates"),
        "constant_columns": eda.get("constant_columns"),
        "top_correlations": eda.get("top_correlations"),
        "rare_levels": {k: len(v) for k, v in (eda.get("rare_levels") or {}).items()},
        "time_columns": eda.get("time_columns") or eda.get("time_like_candidates"),
        "recommendations": eda.get("recommendations"),
        "question": manifest.get("question"),
        "context": manifest.get("context"),
        "job_id": manifest.get("job_id"),
        "manifest": manifest,
    }


def plan_with_router(ctx: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    fallback = {"plan": ["feature_expert", "modeling", "evaluation"], "decisions": {"budget": "normal"}, "source": "fallback"}
    if not api_key or OpenAI is None:
        if ctx.get("job_id"):
            model_decision(str(ctx.get("job_id")), "Router: OpenAI unavailable or no API key; using fallback")
        return fallback
    try:
        client = OpenAI()
        prompt = {
            "role": "user",
            "content": f"Return JSON with keys plan[list], decisions[object] given context: {json.dumps(ctx)[:4000]}"
        }
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a data analysis planner. Reply JSON only."}, prompt],
            temperature=0.1
        )
        txt = r.choices[0].message.content or "{}"
        sanitized = txt.replace("\n"," ")[:1000]
        job_id = str(ctx.get("job_id") or "")
        if job_id:
            model_decision(job_id, f"Router raw plan: {sanitized}")
        plan = json.loads(txt)
        plan["source"] = "openai"
        return plan
    except Exception as e:
        if ctx.get("job_id"):
            model_decision(str(ctx.get("job_id")), f"Router failed: {e}; using fallback")
        return fallback

