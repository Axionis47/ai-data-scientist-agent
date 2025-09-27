"""Evaluation Service

Utilities for computing metrics and choosing thresholds, used by modeling.

- compute_binary_metrics(y_true, proba, preds=None) -> dict
  returns {acc, f1, roc_auc, pr_auc, best_f1_threshold}
- choose_threshold(y_true, proba, strategy="f1") -> (thr, metric_value)

Notes
- Functions are conservative and handle edge cases gracefully.
- If proba is unavailable, falls back to using predicted labels where possible.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore


def choose_threshold(y_true, proba, strategy: str = "f1") -> Tuple[float, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    # Default strategy: maximize F1 on PR curve
    try:
        prec, rec, thr = precision_recall_curve(y_true, proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        if len(thr) == 0:
            return 0.5, float(np.nan)
        best_idx = int(np.argmax(f1s[:-1]))  # last f1 corresponds to threshold=1 sentinel
        return float(thr[best_idx]), float(f1s[best_idx])
    except Exception:
        return 0.5, float("nan")


def compute_binary_metrics(y_true, proba: Optional[np.ndarray], preds: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        if preds is not None:
            out["acc"] = float(accuracy_score(y_true, preds))
            out["f1"] = float(f1_score(y_true, preds))
    except Exception:
        pass
    try:
        if proba is not None:
            if roc_auc_score is not None:
                out["roc_auc"] = float(roc_auc_score(y_true, proba))
            out["pr_auc"] = float(average_precision_score(y_true, proba))
            thr, best_f1 = choose_threshold(y_true, proba, strategy="f1")
            out["best_f1_threshold"] = float(thr)
            if np.isfinite(best_f1):
                out.setdefault("f1", float(best_f1))
    except Exception:
        pass
    return out




def compute_multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    """Basic multiclass metrics.
    Returns weighted F1 and accuracy; designed as a light wrapper to keep parity with binary.
    """
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import f1_score, accuracy_score
        out["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
        out["acc"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        pass
    return out


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Basic regression metrics (r2, rmse)."""
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import r2_score, root_mean_squared_error
        out["r2"] = float(r2_score(y_true, y_pred))
        try:
            out["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        except Exception:
            import numpy as _np
            out["rmse"] = float(_np.sqrt(_np.mean(((_np.asarray(y_true) - _np.asarray(y_pred)) ** 2))))
    except Exception:
        pass
    return out
