"""Data Quality & Leakage Guardrails Service

Conservative, fast checks that surface common issues before modeling.
Intended to guide the router and inform users; never raises by default.

Checks
- Target viability: class prevalence or regression variance
- Identifier columns: near-unique columns and ID-like names
- Near-perfect predictors: obvious leakage signals (binary/regression)

Outputs
- report = { issues: [ {id, severity, detail} ], recommendations: [str], summary: str }

Notes
- Designed to be cheap; avoid heavy cross-features scans.
- Strict thresholds chosen to minimize false positives; adjust as needed.
"""
from __future__ import annotations
from typing import Any, Dict, List
import math
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore


def _is_binary_series(s: pd.Series) -> bool:
    try:
        vals = pd.Series(s.dropna().unique())
        return len(vals) == 2
    except Exception:
        return False


def assess_target_viability(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"issues": [], "notes": []}
    if target not in df.columns:
        out["issues"].append({"id": "target_missing", "severity": "error", "detail": f"Target '{target}' not in columns"})
        return out
    y = df[target]
    n = len(y)
    if n == 0:
        out["issues"].append({"id": "no_rows", "severity": "error", "detail": "No rows in dataset"})
        return out
    if ptypes.is_numeric_dtype(y):
        # Regression viability: non-zero variance
        try:
            var = float(np.nanvar(y.values))
        except Exception:
            var = 0.0
        if not (var > 0.0):
            out["issues"].append({"id": "zero_variance_target", "severity": "error", "detail": "Regression target variance is zero"})
        else:
            out["notes"].append(f"Regression target variance={var:.4g}")
    else:
        # Classification viability: binary prevalence
        if _is_binary_series(y):
            p = float(np.mean(pd.to_numeric(y, errors="coerce"))) if ptypes.is_numeric_dtype(y) else float(y.astype("category").cat.codes.mean())
            minority = min(p, 1 - p)
            out["notes"].append(f"Binary prevalence minority={minority:.3f}")
            if minority < 0.01:
                out["issues"].append({"id": "extreme_imbalance", "severity": "warn", "detail": f"Minority class <1% ({minority:.3f})"})
        else:
            # Multi-class: require minimal per-class counts
            counts = y.value_counts(dropna=False)
            minc = int(counts.min()) if len(counts) else 0
            if minc < 5:
                out["issues"].append({"id": "tiny_class", "severity": "warn", "detail": f"At least one class has <5 samples ({minc})"})
    return out


def detect_identifier_columns(df: pd.DataFrame, nunique: Dict[str, int] | None = None) -> List[str]:
    n = len(df)
    nunique_map = nunique or {c: int(df[c].nunique(dropna=False)) for c in df.columns}
    id_like = []
    for c, u in nunique_map.items():
        name = str(c).lower()
        if u >= max(100, int(0.95 * max(1, n))):
            id_like.append(c)
            continue
        if re.search(r"\b(id|uuid|guid|hash)\b", name):
            if u >= int(0.9 * max(1, n)):
                id_like.append(c)
    return id_like


def detect_near_perfect_predictors(df: pd.DataFrame, target: str, max_check_cols: int = 50) -> List[str]:
    if target not in df.columns:
        return []
    y = df[target]
    cols = [c for c in df.columns if c != target]
    # Limit to first N non-null columns to keep it cheap
    cols = [c for c in cols if df[c].notna().sum() > 0][:max_check_cols]
    flagged: List[str] = []
    if _is_binary_series(y):
        # Numeric: AUC close to 1.0/0.0; Categorical: category-pure mapping covering most rows
        for c in cols:
            s = df[c]
            try:
                if ptypes.is_numeric_dtype(s) and roc_auc_score is not None:
                    # Cast y to 0/1 if needed
                    yy = y
                    if not ptypes.is_numeric_dtype(yy):
                        yy = (yy.astype("category").cat.codes > 0).astype(int)
                    score = roc_auc_score(yy, s)
                    if score is not None and (score > 0.995 or score < 0.005):
                        flagged.append(c)
                else:
                    # Categorical purity: if most frequent categories are near-pure to one class
                    vc = s.astype("category").value_counts(dropna=False)
                    top_levels = vc.index[: min(20, len(vc))]
                    covered = 0
                    pure_hits = 0
                    for lvl in top_levels:
                        mask = s == lvl
                        ct = int(mask.sum())
                        if ct < 5:
                            continue
                        covered += ct
                        mean_y = float(pd.to_numeric(y[mask], errors="coerce").fillna(0).mean()) if ptypes.is_numeric_dtype(y) else float((y[mask].astype("category").cat.codes > 0).mean())
                        if mean_y < 0.01 or mean_y > 0.99:
                            pure_hits += ct
                    if covered > 0 and (pure_hits / covered) > 0.98 and covered / max(1, len(y)) > 0.5:
                        flagged.append(c)
            except Exception:
                continue
    else:
        # Regression: Pearson correlation close to 1
        for c in cols:
            s = df[c]
            try:
                if ptypes.is_numeric_dtype(s) and ptypes.is_numeric_dtype(y):
                    corr = float(pd.Series(s).corr(pd.Series(y)))
                    if not math.isnan(corr) and abs(corr) > 0.999:
                        flagged.append(c)
            except Exception:
                continue
    return flagged


def data_quality_report(job_id: str, df: pd.DataFrame, eda: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    target = (manifest.get("framing") or {}).get("target") or manifest.get("target") or eda.get("target")
    nunique = eda.get("nunique") or {}
    issues: List[Dict[str, Any]] = []
    recs: List[str] = []

    if target:
        tviab = assess_target_viability(df, target)
        issues.extend(tviab.get("issues", []))
    else:
        issues.append({"id": "target_unknown", "severity": "info", "detail": "Target not set yet; viability not assessed"})

    id_cols = detect_identifier_columns(df, nunique)
    if id_cols:
        issues.append({"id": "identifier_columns", "severity": "warn", "detail": f"Identifier-like columns detected: {id_cols[:5]}"})
        recs.append("Consider excluding ID-like columns from modeling features to avoid leakage.")

    if target:
        leak_cols = detect_near_perfect_predictors(df, target)
        if leak_cols:
            issues.append({"id": "near_perfect_predictors", "severity": "warn", "detail": f"Potential leakage via columns: {leak_cols[:5]} (near-perfect mapping to target)"})
            recs.append("Investigate flagged columns; consider removing or shifting to prevent leakage.")

    # Timeseries note if time columns present and split is time
    time_cols = eda.get("time_columns") or eda.get("time_like_candidates") or []
    dec = ((manifest.get("router_plan") or {}).get("decisions") or {})
    if time_cols and str(dec.get("split") or "").lower() == "time":
        recs.append("Using time-based split; ensure features do not include future information relative to the split.")

    summary = "; ".join(sorted(set([i.get("id") for i in issues]))) or "OK"
    return {"issues": issues, "recommendations": recs, "summary": summary}


def summarize_outliers(df: pd.DataFrame, max_cols: int = 10) -> Dict[str, Any]:
    """Summarize potential outliers per numeric column using an IQR rule.
    Returns a dict {col: {q1, q3, iqr, lower, upper, n_outliers}} for up to max_cols.
    """
    out: Dict[str, Any] = {}
    try:
        num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])][: max(0, int(max_cols))]
        for c in num_cols:
            try:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) < 10:
                    continue
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 - 1.5 * iqr + 3.0 * iqr  # keep symmetrical expression clear
                upper = q3 + 1.5 * iqr
                n_out = int(((s < lower) | (s > upper)).sum())
                out[c] = {"q1": q1, "q3": q3, "iqr": float(iqr), "lower": lower, "upper": upper, "n_outliers": n_out}
            except Exception:
                continue
    except Exception:
        return {}
    return out

