"""Encoding Service

Contains supervised encoders that are safe to use inside scikit-learn Pipelines.
Currently provides a target mean encoder with optional out-of-fold encoding for
training data to reduce overfitting.

Classes
- TargetMeanEncoder(cols=None, smoothing=10.0, oof_folds=5)
  * For binary classification and regression
  * Returns a dense numpy array with one encoded column per input column

Notes
- For binary classification, the target is assumed 0/1 (if not numeric, a 0/1
  coding is inferred from category codes >0).
- For regression, the target mean is used directly.
- For multiclass, this encoder is not applied; callers should avoid wiring it.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, smoothing: float = 10.0, oof_folds: int = 5, random_state: int = 42):
        self.cols = cols
        self.smoothing = float(smoothing)
        self.oof_folds = int(oof_folds)
        self.random_state = int(random_state)
        # Fitted attributes
        self._prior_: float | None = None
        self._maps_: Dict[str, Dict[Any, float]] = {}
        self._is_binary_: bool | None = None
        self._trained_cols_: List[str] = []
        self._oof_buffer_: Optional[np.ndarray] = None

    def _as_numeric_target(self, y: pd.Series) -> pd.Series:
        if ptypes.is_numeric_dtype(y):
            # Binary if only two unique values
            if y.nunique(dropna=True) <= 2:
                self._is_binary_ = True
            else:
                self._is_binary_ = False
            return y.astype(float)
        # Non-numeric: try to coerce to 0/1 for binary
        codes = y.astype("category").cat.codes
        # Mark -1 (NaN) as 0
        codes = codes.where(codes >= 0, other=0)
        self._is_binary_ = bool(len(pd.unique(codes)) <= 2)
        # Map to 0/1
        return (codes > 0).astype(float)

    def fit(self, X, y):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        y = pd.Series(y)
        if self.cols is None:
            self._trained_cols_ = list(df.columns)
        else:
            self._trained_cols_ = list(self.cols)
        yt = self._as_numeric_target(y)
        # Prior mean across training
        self._prior_ = float(np.nanmean(yt.values)) if len(yt) else 0.0
        # Full-data maps for inference
        self._maps_ = {}
        for c in self._trained_cols_:
            try:
                grp = pd.DataFrame({"x": df[c], "y": yt}).groupby("x")["y"].agg(["mean", "count"])  # type: ignore
                mean = grp["mean"].values
                cnt = grp["count"].values
                # Smoothing
                m = (cnt * mean + self.smoothing * self._prior_) / (cnt + self.smoothing)
                self._maps_[c] = {k: float(v) for k, v in zip(grp.index.tolist(), m.tolist())}
            except Exception:
                self._maps_[c] = {}
        # Prepare OOF buffer if requested
        self._oof_buffer_ = None
        if self.oof_folds and self.oof_folds > 1:
            try:
                n = len(df)
                oof = np.zeros((n, len(self._trained_cols_)), dtype=float)
                if self._is_binary_:
                    splitter = StratifiedKFold(n_splits=min(self.oof_folds, max(2, len(df) // 3)), shuffle=True, random_state=self.random_state)
                    split_iter = splitter.split(df, y)
                else:
                    splitter = KFold(n_splits=min(self.oof_folds, max(2, len(df) // 3)), shuffle=True, random_state=self.random_state)
                    split_iter = splitter.split(df)
                for tr_idx, va_idx in split_iter:
                    yt_tr = yt.iloc[tr_idx]
                    df_tr = df.iloc[tr_idx]
                    # Fit maps on training fold only
                    maps_fold: Dict[str, Dict[Any, float]] = {}
                    for j, c in enumerate(self._trained_cols_):
                        try:
                            grp = pd.DataFrame({"x": df_tr[c], "y": yt_tr}).groupby("x")["y"].agg(["mean", "count"])  # type: ignore
                            mean = grp["mean"].values
                            cnt = grp["count"].values
                            m = (cnt * mean + self.smoothing * self._prior_) / (cnt + self.smoothing)
                            maps_fold[c] = {k: float(v) for k, v in zip(grp.index.tolist(), m.tolist())}
                        except Exception:
                            maps_fold[c] = {}
                    # Apply to validation fold
                    for j, c in enumerate(self._trained_cols_):
                        s = df.iloc[va_idx][c]
                        mapped = s.map(maps_fold[c]).fillna(self._prior_).astype(float)
                        oof[va_idx, j] = mapped.values
                self._oof_buffer_ = oof
            except Exception:
                self._oof_buffer_ = None
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X
        n = len(df)
        k = len(self._trained_cols_)
        out = np.zeros((n, k), dtype=float)
        for j, c in enumerate(self._trained_cols_):
            s = df[c]
            mapped = s.map(self._maps_.get(c, {})).fillna(self._prior_).astype(float)
            out[:, j] = mapped.values
        return out

    def fit_transform(self, X, y=None, **fit_params):
        # When y is available and OOF is enabled, return out-of-fold encodings for the training set
        if y is None:
            self.fit(X, y=None)
            return self.transform(X)
        self.fit(X, y)
        if self._oof_buffer_ is not None:
            return self._oof_buffer_
        return self.transform(X)


def split_categorical_by_cardinality(eda: Dict[str, Any], cat_cols: List[str], threshold: int = 50) -> tuple[list[str], list[str]]:
    """Split categorical columns into low/high-cardinality groups using EDA nunique if available.
    Returns (low, high).
    """
    nunique = eda.get("nunique") or {}
    low, high = [], []
    for c in cat_cols:
        try:
            u = int(nunique.get(c)) if c in nunique else None
        except Exception:
            u = None
        if u is None:
            # Unknown -> treat as low by default
            low.append(c)
        elif u > threshold:
            high.append(c)
        else:
            low.append(c)
    return low, high

