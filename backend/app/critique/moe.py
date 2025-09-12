from __future__ import annotations
import os
import json
from typing import Dict, Any, List

# Post-model critique feature flags
CRITIQUE_POST_MODEL = os.getenv("CRITIQUE_POST_MODEL", "false").lower() in (
    "1",
    "true",
    "yes",
)


def _minority_ratio(modeling: Dict[str, Any]) -> float | None:
    try:
        best = modeling.get("best") or {}
        # Optional: expect class balance in modeling['best']['class_balance']
        cb = best.get("class_balance")
        if isinstance(cb, dict) and len(cb) >= 2:
            total = sum(cb.values()) or 1
            frac = min(cb.values()) / total
            return float(frac)
    except Exception:
        return None
    return None


def deterministic_checks(
    eda: Dict[str, Any], modeling: Dict[str, Any]
) -> Dict[str, Any]:
    """Cheap, objective checks to ground LLM experts and provide signals.
    Returns a dict with booleans/values the router can use.
    """
    signals: Dict[str, Any] = {}
    try:
        # Class imbalance
        frac = None
        try:
            frac = _minority_ratio(modeling)
        except Exception:
            frac = None
        if frac is None:
            # Fall back to EDA target balance if present later
            pass
        signals["class_minority_ratio"] = frac

        # Category density & cardinality
        nunique = eda.get("nunique") or {}
        rows = int((eda.get("shape") or [0, 0])[0] or 0)
        high_card_cols = [
            c
            for c, n in nunique.items()
            if isinstance(n, (int, float)) and n > max(50, 0.5 * rows)
        ]
        signals["high_card_cols"] = high_card_cols

        # Time-like signals
        signals["has_time_col"] = bool(
            eda.get("time_columns") or eda.get("time_like_candidates")
        )

        # Overfit proxy (if available): cv vs test metric gap on best model
        best = modeling.get("best") or {}
        cv_mean = best.get("cv_mean")
        test_metric = (
            best.get("f1")
            if modeling.get("task") == "classification"
            else best.get("r2")
        )
        if isinstance(cv_mean, (int, float)) and isinstance(test_metric, (int, float)):
            signals["overfit_gap"] = float(cv_mean - test_metric)
        else:
            signals["overfit_gap"] = None
    except Exception:
        pass
    return signals


def modeling_choices_expert(
    eda: Dict[str, Any], modeling: Dict[str, Any], signals: Dict[str, Any]
) -> Dict[str, Any]:
    """Rule-based first pass expert to avoid LLM costs. Emits normalized output.
    We can later swap internals with a JSON-first LLM call.
    """
    issues: List[Dict[str, Any]] = []
    recs: List[Dict[str, Any]] = []

    # Imbalance handling
    frac = signals.get("class_minority_ratio")
    if (
        modeling.get("task") == "classification"
        and isinstance(frac, (int, float))
        and frac < 0.15
    ):
        issues.append(
            {
                "id": "imbalance_detected",
                "severity": "high",
                "evidence": f"minority={frac:.2f}",
            }
        )
        recs.append({"action": "enable_threshold_tuning", "params": {"metric": "f1"}})
        recs.append({"action": "set_metric", "params": {"primary": "f1"}})
        recs.append({"action": "enable_class_weight", "params": {"mode": "balanced"}})

    # Time-series split suggestion
    if signals.get("has_time_col"):
        issues.append({"id": "time_series_detected", "severity": "medium"})
        recs.append({"action": "set_split", "params": {"strategy": "time"}})

    # High cardinality categoricals
    hc = signals.get("high_card_cols") or []
    if len(hc) > 0:
        issues.append(
            {
                "id": "high_cardinality",
                "severity": "medium",
                "evidence": {"cols": hc[:5]},
            }
        )
        recs.append(
            {
                "action": "enable_high_cardinality",
                "params": {"method": "topk+other", "k": 50},
            }
        )

    # Overfit gap
    gap = signals.get("overfit_gap")
    if isinstance(gap, (int, float)) and gap > 0.1:
        issues.append(
            {"id": "cv_test_gap_large", "severity": "medium", "evidence": {"gap": gap}}
        )

    return {
        "expert": "modeling_choices",
        "issues": issues,
        "recommendations": recs,
        "confidence": 0.7 if issues or recs else 0.3,
    }


def critique_post_model(
    job_id: str,
    eda: Dict[str, Any],
    modeling: Dict[str, Any],
    explain: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Main entry to produce a critique after modeling.
    Currently deterministic + rule-based; can be extended to LLM MoE.
    """
    try:
        signals = deterministic_checks(eda, modeling)
        mc = modeling_choices_expert(eda, modeling, signals)
        summary = []
        if mc.get("issues"):
            ids = ", ".join(i.get("id") for i in mc["issues"])
            summary.append(f"Issues: {ids}.")
        if mc.get("recommendations"):
            acts = ", ".join(a.get("action") for a in mc["recommendations"])
            summary.append(f"Recommend: {acts}.")
        out = {
            "phase": "post_modeling",
            "summary": " ".join(summary) or "No critical issues detected.",
            "recommendations": mc.get("recommendations", []),
            "issues": mc.get("issues", []),
            "confidence": mc.get("confidence", 0.5),
        }
        return out
    except Exception as e:
        return {"phase": "post_modeling", "error": str(e)}
