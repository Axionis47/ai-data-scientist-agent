from __future__ import annotations
from typing import Dict, Any

from ..core.logs import model_decision
from ..core.config import SAFE_AUTO_ACTIONS


def apply_safe_actions(
    job_id: str, eda: Dict[str, Any], modeling: Dict[str, Any], framing: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply conservative, safe auto-actions to framing based on EDA and modeling signals.
    Currently:
      - If classification with imbalance (minority < 20%), set class_weight='balanced' for logistic
      - Ensure stratified split flag is set (informational, modeling already stratifies)
    """
    if not SAFE_AUTO_ACTIONS:
        return framing
    try:
        task = modeling.get("task") or (
            "classification" if (eda.get("target_type") == "categorical") else None
        )
        if task != "classification":
            return framing
        target = framing.get("target") or eda.get("target")
        if not target:
            return framing
        class_counts = (eda.get("target_counts") or {}).get(str(target)) or eda.get(
            "class_counts"
        )
        if isinstance(class_counts, dict) and sum(class_counts.values()) > 0:
            total = sum(class_counts.values())
            minority = min(class_counts.values()) / total
            if minority < 0.2:
                framing.setdefault("hints", {})["class_weight"] = "balanced"
                model_decision(
                    job_id, "Auto-action: set class_weight=balanced due to imbalance"
                )
        framing.setdefault("hints", {})["stratified"] = True
    except Exception:
        pass
    return framing
