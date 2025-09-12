from __future__ import annotations
from typing import Dict, Any, List


def validate_modeling(modeling: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(modeling, dict):
        return ["modeling is not a dict"]
    task = modeling.get("task")
    if task not in ("classification", "regression"):
        errs.append("task missing or invalid")
    if not isinstance(modeling.get("leaderboard"), list):
        errs.append("leaderboard missing or not a list")
    if not isinstance(modeling.get("best"), dict):
        errs.append("best missing or not a dict")
    feats = modeling.get("features")
    if not (isinstance(feats, dict) and isinstance(feats.get("numeric"), int) and isinstance(feats.get("categorical"), int)):
        errs.append("features missing or invalid")
    tools = modeling.get("selected_tools")
    if not (isinstance(tools, list)):
        errs.append("selected_tools missing or invalid")
    return errs


def validate_eda(eda: Any) -> List[str]:
    if not isinstance(eda, dict):
        return ["eda is not a dict"]
    keys = ("shape", "missing")
    miss = [k for k in keys if k not in eda]
    return [f"eda missing key: {k}" for k in miss]


# Simple JSON-first report schema
# Expected shape:
# {
#   "title": str,
#   "kpis": {str: number},
#   "sections": [{"heading": str, "items": [str]}]  OR  [{"heading": str, "html": str}],
#   "model_card"?: {"name": str, "task": str, "metric_primary": str, "metric_value": number, "features": {"numeric": int, "categorical": int}, "candidates": [str], "threshold"?: number}
# }

def validate_report_json(obj: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["report JSON is not a dict"]
    if not isinstance(obj.get("title"), str):
        errs.append("title missing or not a string")
    kpis = obj.get("kpis")
    if not isinstance(kpis, dict):
        errs.append("kpis missing or not a dict")
    secs = obj.get("sections")
    if not isinstance(secs, list):
        errs.append("sections missing or not a list")
    else:
        for i, s in enumerate(secs):
            if not isinstance(s, dict):
                errs.append(f"section[{i}] not a dict")
                continue
            if not isinstance(s.get("heading"), str):
                errs.append(f"section[{i}].heading missing or not string")
            if ("items" in s) and not isinstance(s.get("items"), list):
                errs.append(f"section[{i}].items not a list")
            if ("html" in s) and not isinstance(s.get("html"), str):
                errs.append(f"section[{i}].html not a string")
    # Optional model_card validation
    mc = obj.get("model_card")
    if mc is not None:
        if not isinstance(mc, dict):
            errs.append("model_card not a dict")
        else:
            for key in ("name", "task", "metric_primary", "metric_value"):
                if key not in mc:
                    errs.append(f"model_card missing {key}")
            feats = mc.get("features")
            if feats is not None and not (isinstance(feats, dict)):
                errs.append("model_card.features invalid")
            cand = mc.get("candidates")
            if cand is not None and not isinstance(cand, list):
                errs.append("model_card.candidates invalid")
    return errs

