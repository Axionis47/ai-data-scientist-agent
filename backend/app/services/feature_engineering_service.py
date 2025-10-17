"""Feature Engineering Service

Lightweight, safe feature enrichments applied prior to modeling.
Currently implements datetime-derived features.

Contract
- add_datetime_features(df, eda, manifest) -> (df_out, report)
  - Adds year/month/dayofweek derived numeric columns for each detected datetime column
  - Returns a small report listing added columns

Design
- Conservative: no target leakage; derives only from features, not target
- Fast: avoids heavy scans; can be extended later (target encoding, interactions)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandas.api.types as ptypes


def _detect_datetime_columns(df: pd.DataFrame, eda: Dict[str, Any]) -> List[str]:
    cands = set(eda.get("time_columns") or []) | set(
        eda.get("time_like_candidates") or []
    )
    cols: List[str] = []
    for c in df.columns:
        if c in cands:
            cols.append(c)
            continue
        try:
            if ptypes.is_datetime64_any_dtype(df[c]):
                cols.append(c)
        except Exception:
            continue
    return cols


def add_datetime_features(
    df: pd.DataFrame, eda: Dict[str, Any], manifest: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame):
        return df, {"added": [], "notes": ["no_df"]}
    dcols = _detect_datetime_columns(df, eda)
    added: List[str] = []
    out = df.copy()
    for c in dcols:
        try:
            s = pd.to_datetime(out[c], errors="coerce")
            out[f"{c}_year"] = s.dt.year
            out[f"{c}_month"] = s.dt.month
            out[f"{c}_dow"] = s.dt.dayofweek
            added.extend([f"{c}_year", f"{c}_month", f"{c}_dow"])
        except Exception:
            continue
    rep = {"added": added, "base": list(dcols)}
    return out, rep


def add_timeseries_features(
    df: pd.DataFrame,
    eda: Dict[str, Any],
    manifest: Dict[str, Any],
    max_numeric: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add simple, safe time-series features when a time column is detected.

    - Sorts by the detected time column
    - For up to `max_numeric` numeric feature columns (excluding target), adds:
        * lag1: previous value
        * rollX_mean: mean over previous X values for X in windows (each shifted by 1)
    - Leaves NaNs for the first few rows; downstream imputers handle them

    Router hints (optional, via manifest.router_plan.decisions):
    - timeseries_fe: false/off -> disable this step
    - lag_windows: list[int] (e.g., [3, 7]) -> rolling window sizes to use (defaults to [3])

    Returns (df_out, report) where report lists added columns and chosen time_col.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df, {"added": [], "notes": ["no_df"]}

    decisions = ((manifest or {}).get("router_plan") or {}).get("decisions") or {}
    ts_flag = decisions.get("timeseries_fe")
    if isinstance(ts_flag, str) and ts_flag.lower() in (
        "off",
        "disable",
        "disabled",
        "false",
        "0",
    ):
        return df, {"added": [], "notes": ["disabled_by_router"]}
    if ts_flag is False:
        return df, {"added": [], "notes": ["disabled_by_router"]}

    # Determine time column and target
    time_col = None
    dcols = _detect_datetime_columns(df, eda or {})
    if dcols:
        # Prefer the first that exists in the frame
        time_col = next((c for c in dcols if c in df.columns), None)
    target = (manifest.get("framing") or {}).get("target")

    if not time_col or time_col not in df.columns:
        return df, {"added": [], "notes": ["no_time_col"]}

    out = df.copy()
    try:
        ts = pd.to_datetime(out[time_col], errors="coerce")
        order = ts.sort_values(kind="mergesort").index
        out_sorted = out.loc[order].copy()
    except Exception:
        return df, {"added": [], "time_col": time_col, "notes": ["time_parse_failed"]}

    # Select numeric columns excluding the target and the time column
    num_cols = [c for c in out_sorted.columns if ptypes.is_numeric_dtype(out_sorted[c])]
    num_cols = [c for c in num_cols if c != target and c != time_col]
    if not num_cols:
        return out, {"added": [], "time_col": time_col}

    # Limit to first max_numeric to avoid feature explosion
    num_cols = num_cols[: max(1, int(max_numeric))]

    # Windows to use for rolling means
    windows = decisions.get("lag_windows")
    if not isinstance(windows, list) or not windows:
        windows = [3]
    # Sanitize windows
    windows = [int(w) for w in windows if isinstance(w, (int, float)) and int(w) >= 2][
        :3
    ]
    if not windows:
        windows = [3]

    added: List[str] = []
    for c in num_cols:
        try:
            out_sorted[f"{c}_lag1"] = out_sorted[c].shift(1)
            added.append(f"{c}_lag1")
            for w in windows:
                cname = f"{c}_roll{w}_mean"
                out_sorted[cname] = (
                    out_sorted[c].rolling(window=w, min_periods=1).mean().shift(1)
                )
                added.append(cname)
        except Exception:
            continue

    # Restore original order
    out_final = out_sorted.loc[out.index]
    rep = {"added": added, "time_col": time_col, "base": num_cols, "windows": windows}
    return out_final, rep


def add_text_features(
    df: pd.DataFrame,
    eda: Dict[str, Any],
    max_cols: int = 5,
    max_len: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add lightweight text features for short object/categorical columns.

    For up to `max_cols` text-like columns, adds:
      - <col>_len, <col>_num_alpha, <col>_num_digit, <col>_num_space, <col>_num_punct, <col>_num_words
    Applies .astype(str) and clips overly long strings to `max_len` to limit cost.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df, {"added": [], "notes": ["no_df"]}

    # Candidate text columns: object or categorical with modest cardinality
    from pandas.api.types import CategoricalDtype

    cand = []
    for c in df.columns:
        try:
            dt = df[c].dtype
            is_texty = ptypes.is_object_dtype(dt) or isinstance(dt, CategoricalDtype)
            if is_texty and int(pd.Series(df[c]).nunique(dropna=False)) >= 2:
                cand.append(c)
        except Exception:
            continue
    cand = cand[: max(0, int(max_cols))]

    if not cand:
        return df, {"added": [], "notes": ["no_text_cols"]}

    out = df.copy()
    added: List[str] = []
    for c in cand:
        try:
            s = out[c].astype(str).str.slice(0, max_len)
            out[f"{c}_len"] = s.str.len()
            out[f"{c}_num_alpha"] = s.str.count(r"[A-Za-z]")
            out[f"{c}_num_digit"] = s.str.count(r"[0-9]")
            out[f"{c}_num_space"] = s.str.count(r"\s")
            out[f"{c}_num_punct"] = s.str.count(r"[\.,;:!?\-\(\)\[\]{}'\"]")
            out[f"{c}_num_words"] = s.str.count(r"\b\w+\b")
            added.extend(
                [
                    f"{c}_len",
                    f"{c}_num_alpha",
                    f"{c}_num_digit",
                    f"{c}_num_space",
                    f"{c}_num_punct",
                    f"{c}_num_words",
                ]
            )
        except Exception:
            continue

    return out, {"added": added, "base": cand}
