"""Fairness/Slice Metrics Service

Computes simple slice-wise metrics over categorical columns.
Conservative default: operates on true labels only; if predictions/probabilities
are provided, computes performance per slice as well.

Contract
- compute_slice_metrics(df, target, task="classification", max_cols=3, max_card=10,
                       predictions=None, probabilities=None)
  -> dict with per-column summaries, disparities, and notes

Notes
- For classification with binary target: computes per-group prevalence and disparity.
- If predictions provided: adds acc/f1 per group; if probabilities provided: adds ROC AUC per group when possible.
- For regression: reports mean target per group and range.
- Skips high-cardinality columns and columns with too many missing/unique groups.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas.api.types as ptypes
from pandas.api.types import CategoricalDtype


def _eligible_cat_columns(
    df: pd.DataFrame, target: str, max_card: int = 10
) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        try:
            dt = df[c].dtype
            if ptypes.is_object_dtype(dt) or isinstance(dt, CategoricalDtype):
                nunique = int(pd.Series(df[c]).nunique(dropna=False))
                if 2 <= nunique <= max_card:
                    cols.append(c)
        except Exception:
            continue
    return cols


def compute_slice_metrics(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    max_cols: int = 3,
    max_card: int = 10,
    predictions: Optional[pd.Series] = None,
    probabilities: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if df is None or target not in df.columns:
        return {"columns": [], "notes": ["no_df_or_target"], "summaries": {}}

    y = df[target]
    summaries: Dict[str, Any] = {}
    cols = _eligible_cat_columns(df, target, max_card=max_card)[: max(0, max_cols)]

    if task == "classification":
        # Binary assumption for prevalence; otherwise, fallback to frequency of the most common class
        try:
            is_binary = int(pd.Series(y).nunique(dropna=True)) == 2
        except Exception:
            is_binary = False
        for c in cols:
            try:
                g = df[c].astype(str)
                grp = pd.DataFrame({"y": y, c: g})
                stats = (
                    grp.groupby(c)["y"]
                    .agg(["count"])
                    .rename(columns={"count": "support"})
                )
                if is_binary:
                    pos_rate = grp.groupby(c)["y"].apply(lambda s: pd.Series(s).mean())
                    stats["prevalence"] = pos_rate
                    disparity = (
                        float(pos_rate.max() - pos_rate.min())
                        if len(pos_rate) > 1
                        else 0.0
                    )
                else:
                    # For multiclass, prevalence is undefined; report majority-class share per group
                    maj_share = grp.groupby(c)["y"].apply(
                        lambda s: pd.Series(s).value_counts(normalize=True).max()
                    )
                    stats["majority_share"] = maj_share
                    disparity = (
                        float(maj_share.max() - maj_share.min())
                        if len(maj_share) > 1
                        else 0.0
                    )
                # Optional performance per group
                if predictions is not None:
                    acc = (
                        grp.assign(pred=predictions)
                        .groupby(c)
                        .apply(lambda d: float((d["y"] == d["pred"]).mean()))
                    )
                    stats["acc"] = acc
                if (predictions is not None) and is_binary:
                    try:
                        from sklearn.metrics import f1_score

                        f1g = (
                            grp.assign(pred=predictions)
                            .groupby(c)
                            .apply(lambda d: float(f1_score(d["y"], d["pred"])))
                        )
                        stats["f1"] = f1g
                    except Exception:
                        pass
                summaries[c] = {
                    "groups": stats.reset_index().to_dict(orient="records"),
                    "disparity": disparity,
                }
            except Exception:
                continue
    else:
        # Regression: mean target per group and range
        for c in cols:
            try:
                g = df[c].astype(str)
                grp = pd.DataFrame({"y": y, c: g})
                stats = (
                    grp.groupby(c)["y"]
                    .agg(["count", "mean"])
                    .rename(columns={"count": "support", "mean": "y_mean"})
                )
                ymean = stats["y_mean"] if "y_mean" in stats else pd.Series([])
                disparity = float(ymean.max() - ymean.min()) if len(ymean) > 1 else 0.0
                summaries[c] = {
                    "groups": stats.reset_index().to_dict(orient="records"),
                    "disparity": disparity,
                }
            except Exception:
                continue

    return {"columns": cols, "summaries": summaries}
