from typing import Dict, Any, Optional, List

"""Modeling pipeline: trains simple candidates, evaluates, and produces explainability outputs.

Key entry: run_modeling(job_id, df, eda, manifest) -> modeling dict.
- Chooses task based on target dtype and cardinality
- Builds preprocessing (impute + TopK + OHE) and candidates (linear/tree/boosting)
- Applies safe CV folds for tiny data, optional quick hyperparam search
- Writes modeling.json for resumability; returns leaderboard, best, and explain artifacts
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
import pandas.api.types as ptypes
from sklearn.calibration import CalibratedClassifierCV


from ..core.logs import model_decision
from .explainers import add_binary_curves, add_regression_diagnostics, add_shap_beeswarm
from ..core.config import (
    JOBS_DIR,
    EARLY_STOP_SAMPLE,
    HGB_MIN_ROWS,
    CV_FOLDS,
    SHAP_ENABLED,
    SHAP_MAX_ROWS,
    SEARCH_TIME_BUDGET,
    CALIBRATE_ENABLED,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from ..services.evaluation_service import compute_binary_metrics

# Try to import XGBoost (optional dependency)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Neural network support flag (can be disabled via config)
MLP_ENABLED = True


class SafeVarianceThreshold(BaseEstimator, TransformerMixin):
    """VarianceThreshold that falls back to passthrough if all features would be removed."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self._selector = None
        self._passthrough = False

    def fit(self, X, y=None):
        self._selector = VarianceThreshold(threshold=self.threshold)
        try:
            self._selector.fit(X)
            # Check if any features would remain
            if self._selector.get_support().sum() == 0:
                self._passthrough = True
            else:
                self._passthrough = False
        except Exception:
            self._passthrough = True
        return self

    def transform(self, X):
        if self._passthrough:
            return X
        try:
            result = self._selector.transform(X)
            # Double-check we have features
            if result.shape[1] == 0:
                return X
            return result
        except Exception:
            return X

    def get_support(self, indices=False):
        if self._passthrough:
            n_features = getattr(self, '_n_features_in_', None)
            if n_features:
                return np.arange(n_features) if indices else np.ones(n_features, dtype=bool)
            return np.array([])
        return self._selector.get_support(indices=indices)


# Minimal TopK encoder (copied from main to avoid deep deps)
class TopKCategorical:
    def __init__(self, cols=None, k=50):
        self.cols = cols
        self.k = k
        self.keep_: Dict[str, set] = {}
        self._fitted_cols_: List[str] = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        # Store the actual columns we're fitting on (may differ from self.cols if X is numpy array)
        self._fitted_cols_ = list(df.columns)
        cols = self.cols or self._fitted_cols_
        for c in cols:
            if c in df.columns:
                vc = df[c].astype(str).value_counts()
                self.keep_[c] = set(vc.head(self.k).index.tolist())
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        # Handle case where X is numpy array (columns are 0, 1, 2, ...)
        # Map positional indices to original column names if needed
        cols_to_transform = self.cols or self._fitted_cols_
        for c in cols_to_transform:
            if c in df.columns:
                df[c] = df[c].astype(str)
                df[c] = df[c].where(df[c].isin(self.keep_.get(c, set())), other="__OTHER__")
        return df


class FittedEnsemble:
    """Simple prediction-time ensemble over already-fitted estimators.
    For binary classification: soft-average predict_proba when available; else majority vote.
    For regression: average predictions.
    """

    def __init__(self, members: Dict[str, Any], task: str = "binary"):
        self.members = members
        self.task = task

    def predict_proba(self, X):
        if self.task != "binary":
            raise AttributeError("predict_proba not available for non-binary ensemble")
        probs = []
        for m in self.members.values():
            if hasattr(m, "predict_proba"):
                try:
                    p = m.predict_proba(X)[:, 1]
                    probs.append(p)
                except Exception:
                    continue
        if not probs:
            raise AttributeError("No member provides predict_proba")

        return np.vstack([1 - np.mean(probs, axis=0), np.mean(probs, axis=0)]).T

    def predict(self, X):
        if self.task == "binary":
            # Try soft voting
            if any(hasattr(m, "predict_proba") for m in self.members.values()):
                try:
                    proba = self.predict_proba(X)[:, 1]
                    return (proba >= 0.5).astype(int)
                except Exception:
                    pass
            # Fallback majority vote
            preds = []
            for m in self.members.values():
                try:
                    preds.append(m.predict(X))
                except Exception:
                    continue
            if not preds:
                raise RuntimeError("No predictions available from ensemble members")
            preds = np.vstack(preds)
            # Majority vote across rows
            return (preds.mean(axis=0) >= 0.5).astype(int)
        else:
            # Regression average
            preds = []
            for m in self.members.values():
                try:
                    preds.append(m.predict(X))
                except Exception:
                    continue
            if not preds:
                raise RuntimeError("No predictions available from ensemble members")
            return np.mean(np.vstack(preds), axis=0)


class StackingEnsemble:
    """Meta-learner over base estimators.
    Binary: meta is LogisticRegression on base predict_proba.
    Regression: meta is LinearRegression on base predictions.
    """

    def __init__(self, members: Dict[str, Any], meta: Any, task: str = "binary"):
        self.members = members
        self.meta = meta
        self.task = task

    def _make_features(self, X):
        feats = []
        for m in self.members.values():
            if self.task == "binary":
                if hasattr(m, "predict_proba"):
                    try:
                        feats.append(m.predict_proba(X)[:, 1])
                    except Exception:
                        feats.append(m.predict(X))
                else:
                    feats.append(m.predict(X))
            else:
                feats.append(m.predict(X))
        return np.vstack(feats).T

    def predict_proba(self, X):
        if self.task != "binary":
            raise AttributeError("predict_proba only for binary task")

        Z = self._make_features(X)
        if hasattr(self.meta, "predict_proba"):
            p = self.meta.predict_proba(Z)[:, 1]
        else:
            # Fallback via decision_function sigmoid
            try:
                from scipy.special import expit

                p = expit(self.meta.decision_function(Z))
            except Exception:
                p = self.meta.predict(Z)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        Z = self._make_features(X)
        if self.task == "binary":
            if hasattr(self.meta, "predict_proba"):
                p = self.meta.predict_proba(Z)[:, 1]
                return (p >= 0.5).astype(int)
            try:
                from scipy.special import expit

                p = expit(self.meta.decision_function(Z))
                return (p >= 0.5).astype(int)
            except Exception:
                return (self.meta.predict(Z) >= 0.5).astype(int)
        else:
            return self.meta.predict(Z)


def quick_search(
    name: str, pipe: Pipeline, X, y, is_class: bool, time_budget: int | None = None
) -> Pipeline:
    import time

    budget = SEARCH_TIME_BUDGET if time_budget is None else int(time_budget)

    if budget <= 0:
        return pipe
    t0 = time.time()
    try:
        if name.startswith("rf"):
            grid = {"est__n_estimators": [100, 200], "est__max_depth": [None, 10, 20]}
        elif name.startswith("hgb"):
            grid = {"est__max_depth": [None, 6, 12], "est__learning_rate": [0.05, 0.1]}
        elif name.startswith("logreg"):
            grid = {"est__C": [0.5, 1.0, 2.0]}
        elif name.startswith("xgb"):
            grid = {"est__max_depth": [4, 6, 8], "est__learning_rate": [0.05, 0.1, 0.2]}
        else:
            grid = {}
        if not grid:
            return pipe
        n_iter = min(4, max(1, sum(len(v) for v in grid.values())))
        # Scale iterations loosely with provided time budget (seconds-ish)
        try:
            n_iter = min(10, max(1, int(budget)))
        except Exception:
            n_iter = min(4, max(1, sum(len(v) for v in grid.values())))
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=n_iter,
            scoring=("f1" if is_class else "r2"),
            cv=(
                StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                if is_class
                else CV_FOLDS
            ),
            n_jobs=-1,
            random_state=42,
        )
        # Simple time budget: cap by iterations via n_iter already small; rely on outer budget guard
        search.fit(X, y)
        elapsed = time.time() - t0
        model_decision(
            "",
            f"Quick search {name}: best={getattr(search,'best_score_',None)} in {elapsed:.1f}s",
        )
        return search.best_estimator_
    except Exception as e:
        model_decision("", f"Quick search failed for {name}: {e}")
        return pipe


def run_modeling(
    job_id: str, df: pd.DataFrame, eda: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # Prepare target and features
    target = manifest.get("target") or eda.get("target")
    if target is None or target not in df.columns:
        raise ValueError("Target not found")
    y = df[target]
    X = df.drop(columns=[target])

    # Drop identifier columns (high-cardinality unique columns that would cause issues)
    # SAFEGUARD: Never drop ALL feature columns - always keep at least some features
    try:
        from ..services.data_quality_service import detect_identifier_columns
        nunique_map = eda.get("nunique") or {}
        id_cols = detect_identifier_columns(X, nunique_map)
        if id_cols:
            # Calculate how many columns would remain after dropping
            remaining_cols = [c for c in X.columns if c not in id_cols]
            if len(remaining_cols) >= 1:
                # Safe to drop - we'll have at least 1 feature left
                X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")
                model_decision(job_id, f"Dropped identifier columns: {id_cols}")
            else:
                # Would drop ALL columns - skip dropping and log warning
                model_decision(job_id, f"SKIPPED dropping identifier columns {id_cols} - would leave 0 features. Keeping all columns.")
    except Exception as e:
        model_decision(job_id, f"Identifier column detection failed: {e}")

    # Columns
    num_cols = [c for c in X.columns if ptypes.is_numeric_dtype(X[c])]
    from pandas.api.types import CategoricalDtype

    cat_cols = [
        c
        for c in X.columns
        if (ptypes.is_object_dtype(X[c]) or isinstance(X[c].dtype, CategoricalDtype))
    ]
    n_rows = len(X)
    cat_ratio = len(cat_cols) / max(1, len(X.columns))
    # Router/manifest decisions (planning) mapped into modeling behavior
    decisions = ((manifest or {}).get("router_plan") or {}).get("decisions") or {}
    framing = (manifest or {}).get("framing") or {}
    desired_metric = (framing.get("metric") or decisions.get("metric") or "").lower()
    # Map router budget to a small per-run search budget (seconds-ish; heuristic via n_iter)
    budget_map = {"low": 0, "normal": 4, "high": 10}
    search_budget = budget_map.get(str(decisions.get("budget") or "").lower())
    # Class weight support for classifiers
    class_weight = str(decisions.get("class_weight") or "").lower()
    class_weight = "balanced" if class_weight in ("balanced", "auto") else None

    # Auto-detect class imbalance and apply balanced weights if not specified
    # Use y directly since it's already defined above
    try:
        if y.nunique(dropna=True) <= 10:  # Classification
            vc = y.value_counts(normalize=True)
            imbalance_ratio = vc.max() / max(vc.min(), 0.001)
            if imbalance_ratio > 3 and class_weight is None:  # More than 3:1 imbalance
                class_weight = "balanced"
                model_decision(job_id, f"Auto-detected class imbalance (ratio={imbalance_ratio:.1f}), using balanced class weights")
    except Exception:
        pass

    # For regression: detect and handle extreme outliers in target
    # Use IQR-based clipping to reduce impact of extreme outliers
    is_regression_task = ptypes.is_numeric_dtype(y) and y.nunique(dropna=True) > 10
    if is_regression_task:
        try:
            q1, q3 = y.quantile(0.25), y.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr  # 3x IQR is very conservative
            upper_bound = q3 + 3 * iqr
            outlier_count = ((y < lower_bound) | (y > upper_bound)).sum()
            if outlier_count > 0:
                # Clip extreme outliers instead of removing them
                y = y.clip(lower=lower_bound, upper=upper_bound)
                model_decision(job_id, f"Clipped {outlier_count} extreme outliers in target (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
        except Exception:
            pass

    # Split strategy (prefer time-based if requested or time columns exist)
    split_pref = str(decisions.get("split") or "").lower()
    time_cols = eda.get("time_columns") or eda.get("time_like_candidates") or []
    time_col = next((c for c in time_cols if c in df.columns), None)
    use_time_split = (split_pref == "time") or bool(time_col)
    try:
        model_decision(
            job_id,
            f"Applied router decisions: metric={desired_metric or 'default'}, budget={decisions.get('budget')}, class_weight={'balanced' if class_weight else 'none'}, split={'time' if use_time_split else 'random'}",
        )
    except Exception:
        pass

    # Preprocess
    # Determine task type early for preprocessing decisions
    is_class = (not ptypes.is_numeric_dtype(y)) or (y.nunique(dropna=True) <= 10)
    is_binary = bool(pd.Series(y).nunique(dropna=True) == 2)

    # Split categorical columns by cardinality (use EDA nunique if available)
    try:
        from ..services.encoding_service import (
            split_categorical_by_cardinality,
            TargetMeanEncoder,
        )

        cat_low, cat_high = split_categorical_by_cardinality(eda or {}, cat_cols)
    except Exception:
        cat_low, cat_high = cat_cols, []

    topk = TopKCategorical(cols=cat_low, k=50)
    cat_low_pipe = Pipeline(
        steps=[
            ("topk", topk),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = [
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat_low", cat_low_pipe, cat_low),
    ]
    # Add target encoding branch only for binary classification or regression
    if is_binary or ptypes.is_numeric_dtype(y):
        try:
            te_pipe = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "te",
                        TargetMeanEncoder(cols=cat_high, smoothing=10.0, oof_folds=5),
                    ),
                ]
            )
            transformers.append(("cat_high_te", te_pipe, cat_high))
        except Exception:
            pass
    pre = ColumnTransformer(transformers, remainder="drop")

    # Feature selection for linear
    use_mi = False
    try:
        Xs = X.head(min(1000, len(X)))
        from ..services.encoding_service import split_categorical_by_cardinality

        cat_low_prev, cat_high_prev = split_categorical_by_cardinality(
            eda or {}, cat_cols
        )
        pre_preview = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), num_cols),
                (
                    "cat_low",
                    Pipeline(
                        steps=[
                            ("topk", TopKCategorical(cols=cat_low_prev, k=50)),
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "ohe",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    cat_low_prev,
                ),
            ],
            remainder="drop",
        )
        pre_preview.fit_transform(Xs)
        # OHE features for low-card + numeric feature count known; add one per high-card col
        try:
            ohe = getattr(pre_preview, "transformers_", [])[1][1].named_steps["ohe"]
            ohe_count = int(ohe.get_feature_names_out().shape[0])
        except Exception:
            ohe_count = 0
        n_feat = int(ohe_count + len(num_cols) + len(cat_high_prev))
        use_mi = n_feat > 100
    except Exception:
        pass

    # Use SafeVarianceThreshold to avoid dropping all features + StandardScaler for linear models
    linear_sel_steps = [
        ("var", SafeVarianceThreshold(0.01)),
        ("scale", StandardScaler()),  # Scale features to prevent overflow in linear models
    ]
    if use_mi:
        if is_class:
            linear_sel_steps.append(("mi", SelectKBest(mutual_info_classif, k=50)))
        else:
            linear_sel_steps.append(("mi", SelectKBest(mutual_info_regression, k=50)))
        model_decision(
            job_id, "Feature selection: SafeVarianceThreshold + StandardScaler + SelectKBest for linear models"
        )
    else:
        model_decision(
            job_id, "Feature selection: SafeVarianceThreshold + StandardScaler for linear models"
        )

    # Candidates
    if is_class:
        cands = [
            (
                "logreg",
                Pipeline(
                    [("prep", pre)]
                    + linear_sel_steps
                    + [
                        (
                            "est",
                            LogisticRegression(max_iter=500, class_weight=class_weight, solver='lbfgs'),
                        )
                    ]
                ),
            ),
            (
                "rf_clf",
                Pipeline(
                    [
                        ("prep", pre),
                        (
                            "est",
                            RandomForestClassifier(
                                n_estimators=200, n_jobs=-1, class_weight=class_weight
                            ),
                        ),
                    ]
                ),
            ),
            (
                "hgb_clf",
                Pipeline(
                    [
                        ("prep", pre),
                        ("est", HistGradientBoostingClassifier(random_state=42)),
                    ]
                ),
            ),
        ]
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            # Calculate scale_pos_weight for imbalanced data
            try:
                pos_count = (y == 1).sum() if is_binary else 1
                neg_count = (y == 0).sum() if is_binary else 1
                scale_pos_weight = neg_count / max(pos_count, 1) if is_binary else 1
            except Exception:
                scale_pos_weight = 1
            cands.append((
                "xgb_clf",
                Pipeline(
                    [
                        ("prep", pre),
                        ("est", XGBClassifier(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.1,
                            scale_pos_weight=scale_pos_weight if is_binary else 1,
                            random_state=42,
                            n_jobs=-1,
                            eval_metric='logloss',
                        )),
                    ]
                ),
            ))
            model_decision(job_id, f"Added XGBoost classifier (scale_pos_weight={scale_pos_weight:.2f})")

        # Add MLP (neural network) classifier if enabled and dataset is large enough
        if MLP_ENABLED and n_rows >= 500:
            cands.append((
                "mlp_clf",
                Pipeline(
                    [("prep", pre)]
                    + linear_sel_steps  # MLP benefits from scaling
                    + [("est", MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                    ))]
                ),
            ))
            model_decision(job_id, "Added MLP classifier (neural network)")
    else:
        # Regression: use RobustScaler to handle outliers
        regression_linear_steps = [
            ("var", SafeVarianceThreshold(0.01)),
            ("scale", RobustScaler()),  # Robust to outliers
        ]
        if use_mi:
            regression_linear_steps.append(("mi", SelectKBest(mutual_info_regression, k=50)))

        cands = [
            (
                "linreg",
                Pipeline(
                    [("prep", pre)] + regression_linear_steps + [("est", LinearRegression())]
                ),
            ),
            (
                "rf_reg",
                Pipeline(
                    [
                        ("prep", pre),
                        ("est", RandomForestRegressor(n_estimators=200, n_jobs=-1)),
                    ]
                ),
            ),
            (
                "hgb_reg",
                Pipeline(
                    [
                        ("prep", pre),
                        ("est", HistGradientBoostingRegressor(random_state=42)),
                    ]
                ),
            ),
        ]
        # Add XGBoost regressor if available
        if XGBOOST_AVAILABLE:
            cands.append((
                "xgb_reg",
                Pipeline(
                    [
                        ("prep", pre),
                        ("est", XGBRegressor(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                        )),
                    ]
                ),
            ))
            model_decision(job_id, "Added XGBoost regressor")

        # Add MLP (neural network) regressor if enabled and dataset is large enough
        if MLP_ENABLED and n_rows >= 500:
            cands.append((
                "mlp_reg",
                Pipeline(
                    [("prep", pre)]
                    + regression_linear_steps  # MLP benefits from scaling
                    + [("est", MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                    ))]
                ),
            ))
            model_decision(job_id, "Added MLP regressor (neural network)")

    # Intelligent gating
    gated = []
    for name, pipe in cands:
        if n_rows < HGB_MIN_ROWS and name.startswith("hgb"):
            model_decision(
                job_id, f"Auto-gating: drop {name} for n_rows<{HGB_MIN_ROWS}"
            )
            continue
        gated.append((name, pipe))
    cands = gated
    if cat_ratio > 0.8:
        cands = sorted(
            cands,
            key=lambda kv: (
                0 if kv[0].startswith("rf") or kv[0].startswith("hgb") else 1
            ),
        )
        model_decision(
            job_id,
            f"Prioritizing tree models due to high categorical ratio={cat_ratio:.2f}",
        )

    # Split strategy (prefer time-based when indicated)
    from sklearn.model_selection import train_test_split, StratifiedKFold

    if use_time_split and time_col is not None:
        try:
            ts = pd.to_datetime(X[time_col], errors="coerce")
            order = ts.sort_values(kind="mergesort").index
            Xs = X.loc[order]
            ys = y.loc[order]
            split_idx = int(len(Xs) * 0.8)
            Xtr, Xte = Xs.iloc[:split_idx], Xs.iloc[split_idx:]
            ytr, yte = ys.iloc[:split_idx], ys.iloc[split_idx:]
            model_decision(job_id, f"Time-based split on '{time_col}' (80/20)")
        except Exception as e:
            model_decision(
                job_id, f"Time-based split failed ({e}); falling back to random"
            )
            strat = y if is_class else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat
            )
    else:
        strat = y if is_class else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )

    # Choose scoring metric for CV based on desired metric
    if is_class:
        if is_binary and desired_metric in ("roc_auc", "auc", "rocauc"):
            scoring_str = "roc_auc"
        elif is_binary and desired_metric in ("pr_auc", "ap", "average_precision"):
            scoring_str = "average_precision"
        elif desired_metric in ("accuracy", "acc"):
            scoring_str = "accuracy"
        else:
            # Use f1 for binary, f1_weighted for multiclass
            scoring_str = "f1" if is_binary else "f1_weighted"
    else:
        scoring_str = "r2"
    # Choose safe CV folds for tiny datasets; prefer time-series CV when time split is in effect
    if use_time_split:
        try:
            from sklearn.model_selection import TimeSeriesSplit

            eff_folds = min(CV_FOLDS, max(2, len(Xtr) // 5)) if len(Xtr) >= 10 else 2
            cv_iter = TimeSeriesSplit(n_splits=eff_folds).split(Xtr)
            model_decision(
                job_id, f"Using TimeSeriesSplit CV with n_splits={eff_folds}"
            )
        except Exception:
            # Fallback to simple K-fold style matching task type
            if is_class:
                try:
                    counts = ytr.value_counts()
                    min_class = int(counts.min()) if len(counts) else 0
                    eff_folds = (
                        max(2, min(CV_FOLDS, min_class)) if min_class >= 2 else 2
                    )
                except Exception:
                    eff_folds = min(CV_FOLDS, 3)
                try:
                    cv_iter = StratifiedKFold(
                        n_splits=eff_folds, shuffle=True, random_state=42
                    ).split(Xtr, ytr)
                except Exception:
                    cv_iter = eff_folds
            else:
                eff_folds = (
                    min(CV_FOLDS, max(2, len(Xtr) // 5)) if len(Xtr) >= 10 else 2
                )
                cv_iter = eff_folds
    else:
        if is_class:
            try:
                counts = ytr.value_counts()
                min_class = int(counts.min()) if len(counts) else 0
                eff_folds = max(2, min(CV_FOLDS, min_class)) if min_class >= 2 else 2
            except Exception:
                eff_folds = min(CV_FOLDS, 3)
            try:
                cv_iter = StratifiedKFold(
                    n_splits=eff_folds, shuffle=True, random_state=42
                ).split(Xtr, ytr)
            except Exception:
                cv_iter = eff_folds
        else:
            eff_folds = min(CV_FOLDS, max(2, len(Xtr) // 5)) if len(Xtr) >= 10 else 2
            cv_iter = eff_folds

    # Early sample evaluation
    sample_eval = None
    if n_rows > 50_000:
        sample_eval = Xtr.sample(n=min(len(Xtr), EARLY_STOP_SAMPLE), random_state=42)
        y_sample = ytr.loc[sample_eval.index]
        model_decision(
            job_id, f"Large dataset: initial evaluation on sample n={len(sample_eval)}"
        )

    leaderboard = []
    fitted: Dict[str, Any] = {}
    proba_map: Dict[str, Optional[np.ndarray]] = {}

    for name, pipe in cands:
        # Optional early sample evaluation
        if sample_eval is not None:
            try:
                pipe.fit(sample_eval, y_sample)
                preds_s = pipe.predict(sample_eval)
                score_s = (
                    f1_score(y_sample, preds_s)
                    if is_class
                    else r2_score(y_sample, preds_s)
                )
                if (is_class and score_s < 0.3) or ((not is_class) and score_s < 0.0):
                    model_decision(
                        job_id,
                        f"Early stop: filtering out {name} due to weak sample score={score_s:.3f}",
                    )
                    continue
                model_decision(
                    job_id,
                    f"Promising on sample: {name} score={score_s:.3f}, proceeding to CV",
                )
            except Exception as e:
                model_decision(job_id, f"Sample eval failed for {name}: {e}")
        # Quick search (obeys per-run search_budget if provided)
        pipe = quick_search(name, pipe, Xtr, ytr, is_class, search_budget)

        # CV scoring
        if isinstance(cv_iter, int):
            scores = cross_val_score(
                pipe,
                Xtr,
                ytr,
                scoring=scoring_str,
                cv=cv_iter,
                n_jobs=-1,
            )
        else:
            scores = []
            for tr_idx, va_idx in cv_iter:
                pipe.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
                preds_cv = pipe.predict(Xtr.iloc[va_idx])
                if is_class:
                    if scoring_str == "accuracy":
                        s = accuracy_score(ytr.iloc[va_idx], preds_cv)
                    elif is_binary:
                        s = f1_score(ytr.iloc[va_idx], preds_cv)
                    else:
                        # Multiclass: use weighted average
                        s = f1_score(ytr.iloc[va_idx], preds_cv, average="weighted")
                else:
                    s = r2_score(ytr.iloc[va_idx], preds_cv)
                scores.append(s)
            scores = np.array(scores)
        cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        fitted[name] = pipe
        if is_class:
            if is_binary:
                try:
                    proba = pipe.predict_proba(Xte)[:, 1]
                    proba_map[name] = proba
                except Exception:
                    proba = None
                # Use evaluation service
                metrics = compute_binary_metrics(yte, proba, preds)
                f1 = float(metrics.get("f1") or f1_score(yte, preds))
                acc = float(metrics.get("acc") or accuracy_score(yte, preds))
                entry = {
                    "name": name,
                    "f1": float(f1),
                    "acc": float(acc),
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }
                if "roc_auc" in metrics:
                    entry["roc_auc"] = float(metrics["roc_auc"])
                if "pr_auc" in metrics:
                    entry["pr_auc"] = float(metrics["pr_auc"])
                if "best_f1_threshold" in metrics:
                    entry["thr"] = float(metrics["best_f1_threshold"])
                leaderboard.append(entry)
            else:
                f1 = f1_score(yte, preds, average="weighted")
                acc = accuracy_score(yte, preds)
                leaderboard.append(
                    {
                        "name": name,
                        "f1": float(f1),
                        "acc": float(acc),
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                    }
                )
        else:
            r2 = r2_score(yte, preds)
            try:
                rmse = root_mean_squared_error(yte, preds)
            except Exception:
                rmse = float(pd.Series(yte - preds).pow(2).mean() ** 0.5)
            leaderboard.append(
                {
                    "name": name,
                    "r2": float(r2),
                    "rmse": float(rmse),
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }
            )

    # Complexity penalty for very small datasets
    if n_rows < 500:
        for m in leaderboard:
            if m["name"].startswith("hgb") or m["name"].startswith("rf"):
                base = (
                    m.get("cv_mean")
                    or (m.get("f1") if is_class else m.get("r2"))
                    or 0.0
                )
                m["cv_mean"] = float(base) * 0.98
        model_decision(
            job_id,
            "Applied complexity penalty to tree/boosting candidates (n_rows<500)",
        )

    # Simple ensemble over top-K candidates (soft voting for binary; averaging for regression)

    # Stacking (meta-learner) over top-K candidates using OOF predictions
    try:
        K = min(3, len(leaderboard))
        if K >= 2:
            # Determine top models as before
            if is_class:
                if is_binary and desired_metric in ("roc_auc", "auc", "rocauc"):

                    def key_fn(m):
                        return m.get("roc_auc") or 0.0

                    reverse = True
                elif is_binary and desired_metric in (
                    "pr_auc",
                    "ap",
                    "average_precision",
                ):

                    def key_fn(m):
                        return m.get("pr_auc") or 0.0

                    reverse = True
                elif desired_metric in ("accuracy", "acc"):

                    def key_fn(m):
                        return m.get("acc") or 0.0

                    reverse = True
                else:

                    def key_fn(m):
                        return m.get("f1") or 0.0

                    reverse = True
            else:
                if desired_metric == "rmse":

                    def key_fn(m):
                        return (
                            m.get("rmse") if m.get("rmse") is not None else float("inf")
                        )

                    reverse = False
                else:

                    def key_fn(m):
                        return m.get("r2") or -1.0

                    reverse = True
            lb_sorted = sorted(leaderboard, key=key_fn, reverse=reverse)
            top_names = [
                m.get("name")
                for m in lb_sorted[:K]
                if m.get("name") in fitted and m.get("name") != "ensemble"
            ]
            if len(top_names) >= 2:
                from sklearn.base import clone

                # Build OOF predictions matrix for training meta
                Z = np.zeros((len(Xtr), len(top_names)), dtype=float)
                if use_time_split:
                    from sklearn.model_selection import TimeSeriesSplit

                    splitter = TimeSeriesSplit(n_splits=max(2, min(CV_FOLDS, 5)))
                    splits = splitter.split(Xtr)
                else:
                    if is_class:
                        from sklearn.model_selection import StratifiedKFold

                        splitter = StratifiedKFold(
                            n_splits=max(2, min(CV_FOLDS, 5)),
                            shuffle=True,
                            random_state=42,
                        )
                        splits = splitter.split(Xtr, ytr)
                    else:
                        from sklearn.model_selection import KFold

                        splitter = KFold(
                            n_splits=max(2, min(CV_FOLDS, 5)),
                            shuffle=True,
                            random_state=42,
                        )
                        splits = splitter.split(Xtr)
                for tr_idx, va_idx in splits:
                    Xtr_i, Xva_i = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
                    ytr_i = ytr.iloc[tr_idx]
                    for j, n in enumerate(top_names):
                        try:
                            est = clone(fitted[n])
                            est.fit(Xtr_i, ytr_i)
                            if is_class and is_binary and hasattr(est, "predict_proba"):
                                Z[va_idx, j] = est.predict_proba(Xva_i)[:, 1]
                            else:
                                Z[va_idx, j] = est.predict(Xva_i)
                        except Exception:
                            # Leave zeros if any failure
                            pass
                # Fit meta model
                if is_class and is_binary:
                    meta = LogisticRegression(max_iter=1000, class_weight=class_weight)
                elif is_class:
                    meta = LogisticRegression(max_iter=1000)
                else:
                    meta = LinearRegression()
                meta.fit(Z, ytr)
                # Build test matrix from fitted base models
                Z_test = []
                for n in top_names:
                    try:
                        if is_class and is_binary:
                            p = proba_map.get(n)
                            if p is None and hasattr(fitted[n], "predict_proba"):
                                p = fitted[n].predict_proba(Xte)[:, 1]
                            Z_test.append(
                                p if p is not None else fitted[n].predict(Xte)
                            )
                        else:
                            Z_test.append(fitted[n].predict(Xte))
                    except Exception:
                        Z_test.append(np.zeros(len(Xte)))
                Z_test = np.vstack(Z_test).T
                # Evaluate stacking
                if is_class and is_binary:
                    try:
                        p_meta = meta.predict_proba(Z_test)[:, 1]
                    except Exception:
                        try:
                            from scipy.special import expit

                            p_meta = expit(meta.decision_function(Z_test))
                        except Exception:
                            p_meta = meta.predict(Z_test)
                    preds_meta = (p_meta >= 0.5).astype(int)
                    metrics = compute_binary_metrics(yte, p_meta, preds_meta)
                    entry = {
                        "name": "stacking",
                        "f1": float(metrics.get("f1") or f1_score(yte, preds_meta)),
                        "acc": float(
                            metrics.get("acc") or accuracy_score(yte, preds_meta)
                        ),
                        "cv_mean": None,
                        "cv_std": None,
                    }
                    if "roc_auc" in metrics:
                        entry["roc_auc"] = float(metrics["roc_auc"])
                    if "pr_auc" in metrics:
                        entry["pr_auc"] = float(metrics["pr_auc"])
                    leaderboard.append(entry)
                    fitted["stacking"] = StackingEnsemble(
                        {n: fitted[n] for n in top_names}, meta, task="binary"
                    )
                elif not is_class:
                    preds_meta = meta.predict(Z_test)
                    r2 = r2_score(yte, preds_meta)
                    try:
                        rmse = root_mean_squared_error(yte, preds_meta)
                    except Exception:
                        rmse = float(pd.Series(yte - preds_meta).pow(2).mean() ** 0.5)
                    leaderboard.append(
                        {
                            "name": "stacking",
                            "r2": float(r2),
                            "rmse": float(rmse),
                            "cv_mean": None,
                            "cv_std": None,
                        }
                    )
                    fitted["stacking"] = StackingEnsemble(
                        {n: fitted[n] for n in top_names}, meta, task="regression"
                    )
    except Exception:
        pass

    try:
        K = min(3, len(leaderboard))
        if K >= 2:
            # Choose sort key based on desired metric
            if is_class:
                if is_binary and desired_metric in ("roc_auc", "auc", "rocauc"):

                    def key_fn(m):
                        return m.get("roc_auc") or 0.0

                    reverse = True
                elif is_binary and desired_metric in (
                    "pr_auc",
                    "ap",
                    "average_precision",
                ):

                    def key_fn(m):
                        return m.get("pr_auc") or 0.0

                    reverse = True
                elif desired_metric in ("accuracy", "acc"):

                    def key_fn(m):
                        return m.get("acc") or 0.0

                    reverse = True
                else:

                    def key_fn(m):
                        return m.get("f1") or 0.0

                    reverse = True
            else:
                if desired_metric == "rmse":

                    def key_fn(m):
                        return (
                            m.get("rmse") if m.get("rmse") is not None else float("inf")
                        )

                    reverse = False
                else:

                    def key_fn(m):
                        return m.get("r2") or -1.0

                    reverse = True
            lb_sorted = sorted(leaderboard, key=key_fn, reverse=reverse)
            top_names = [
                m.get("name") for m in lb_sorted[:K] if m.get("name") in fitted
            ]
            if is_class and is_binary:
                # Soft average probabilities when available
                probas = []
                used = []
                for n in top_names:
                    p = proba_map.get(n)
                    if p is None:
                        try:
                            est = fitted.get(n)
                            if est is not None and hasattr(est, "predict_proba"):
                                p = est.predict_proba(Xte)[:, 1]
                        except Exception:
                            p = None
                    if p is not None:
                        probas.append(p)
                        used.append(n)
                if len(probas) >= 2:
                    proba_ens = np.mean(np.vstack(probas), axis=0)
                    preds_ens = (proba_ens >= 0.5).astype(int)
                    metrics = compute_binary_metrics(yte, proba_ens, preds_ens)
                    entry = {
                        "name": "ensemble",
                        "f1": float(metrics.get("f1") or f1_score(yte, preds_ens)),
                        "acc": float(
                            metrics.get("acc") or accuracy_score(yte, preds_ens)
                        ),
                        "cv_mean": None,
                        "cv_std": None,
                    }
                    if "roc_auc" in metrics:
                        entry["roc_auc"] = float(metrics["roc_auc"])
                    if "pr_auc" in metrics:
                        entry["pr_auc"] = float(metrics["pr_auc"])
                    leaderboard.append(entry)
                    # Register fitted ensemble for downstream explain/predict
                    fitted["ensemble"] = FittedEnsemble(
                        {n: fitted[n] for n in used}, task="binary"
                    )
                    proba_map["ensemble"] = proba_ens
            elif not is_class:
                preds_list = []
                used = []
                for n in top_names:
                    try:
                        preds_list.append(fitted[n].predict(Xte))
                        used.append(n)
                    except Exception:
                        pass
                if len(preds_list) >= 2:
                    preds_ens = np.mean(np.vstack(preds_list), axis=0)
                    r2 = r2_score(yte, preds_ens)
                    try:
                        rmse = root_mean_squared_error(yte, preds_ens)
                    except Exception:
                        rmse = float(pd.Series(yte - preds_ens).pow(2).mean() ** 0.5)
                    leaderboard.append(
                        {
                            "name": "ensemble",
                            "r2": float(r2),
                            "rmse": float(rmse),
                            "cv_mean": None,
                            "cv_std": None,
                        }
                    )
                    fitted["ensemble"] = FittedEnsemble(
                        {n: fitted[n] for n in used}, task="regression"
                    )
    except Exception:
        pass

    # Select best (respect desired_metric when possible)
    if is_class:
        if is_binary and desired_metric in ("roc_auc", "auc", "rocauc"):
            leaderboard.sort(key=lambda m: (m.get("roc_auc") or 0.0), reverse=True)
        elif is_binary and desired_metric in ("pr_auc", "ap", "average_precision"):
            leaderboard.sort(key=lambda m: (m.get("pr_auc") or 0.0), reverse=True)
        elif desired_metric in ("accuracy", "acc"):
            leaderboard.sort(key=lambda m: (m.get("acc") or 0.0), reverse=True)
        else:
            leaderboard.sort(key=lambda m: (m.get("f1") or 0.0), reverse=True)
    else:
        if desired_metric == "rmse":
            leaderboard.sort(
                key=lambda m: (
                    m.get("rmse") if m.get("rmse") is not None else float("inf")
                )
            )
        else:
            leaderboard.sort(key=lambda m: (m.get("r2") or -1.0), reverse=True)
    best = leaderboard[0] if leaderboard else None

    # Optional probability calibration for classifiers when requested by router and enabled
    try:
        calibrate_req = (
            bool(decisions.get("calibration") or framing.get("calibrate"))
            if is_class
            else False
        )
    except Exception:
        calibrate_req = False
    if is_class and CALIBRATE_ENABLED and calibrate_req and best is not None:
        try:
            chosen_name = best.get("name")
            chosen = fitted.get(chosen_name)
            # Skip calibration for ensemble wrapper
            if chosen_name in ("ensemble", "stacking"):
                model_decision(
                    job_id, "Calibration skipped for ensemble/stacking model"
                )
            elif chosen is not None and hasattr(chosen, "predict_proba"):
                method = str(decisions.get("calibration_method") or "sigmoid").lower()
                if method not in ("sigmoid", "isotonic"):
                    method = "sigmoid"
                calib = CalibratedClassifierCV(
                    base_estimator=chosen,
                    cv=min(3, max(2, int(eff_folds))),
                    method=method,
                )
                calib.fit(Xtr, ytr)
                fitted[chosen_name] = calib
                preds = calib.predict(Xte)
                if is_binary:
                    try:
                        proba = calib.predict_proba(Xte)[:, 1]
                    except Exception:
                        proba = None
                    metrics = compute_binary_metrics(yte, proba, preds)
                    best["f1"] = float(metrics.get("f1") or f1_score(yte, preds))
                    best["acc"] = float(
                        metrics.get("acc") or accuracy_score(yte, preds)
                    )
                    if "roc_auc" in metrics:
                        best["roc_auc"] = float(metrics["roc_auc"])
                    if "pr_auc" in metrics:
                        best["pr_auc"] = float(metrics["pr_auc"])
                else:
                    best["f1"] = float(f1_score(yte, preds, average="weighted"))
                    best["acc"] = float(accuracy_score(yte, preds))
                model_decision(
                    job_id,
                    f"Applied probability calibration (method={method}) to {chosen_name}",
                )
            else:
                model_decision(
                    job_id, "Calibration skipped: chosen model lacks predict_proba"
                )
        except Exception as e:
            model_decision(job_id, f"Calibration failed: {e}")

    # Build explain (basic)
    explain: Dict[str, Any] = {}
    try:
        chosen = fitted[best["name"]] if best else None
        if chosen is not None:
            subset = min(len(Xte), 2000)
            res = permutation_importance(
                chosen,
                Xte.iloc[:subset],
                yte.iloc[:subset],
                n_repeats=5,
                random_state=42,
            )
            explain["importances"] = res.importances_mean.tolist()
            plots_dir = JOBS_DIR / job_id / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            base_url = f"/static/jobs/{job_id}/plots"
            if (
                is_class
                and hasattr(chosen, "predict_proba")
                and len(pd.Series(y).nunique()) == 2
            ):
                add_binary_curves(chosen, Xte, yte, plots_dir, base_url, explain)
            if not is_class:
                add_regression_diagnostics(
                    chosen, Xte, yte, plots_dir, base_url, explain
                )
            # SHAP optional
            if (
                SHAP_ENABLED
                and len(Xtr) <= SHAP_MAX_ROWS
                and is_class
                and hasattr(chosen, "predict_proba")
            ):
                try:
                    add_shap_beeswarm(chosen, Xtr, Xte, plots_dir, base_url, explain)
                except Exception:
                    pass
    except Exception:
        pass

    # Attach test-fold predictions for downstream slice metrics (best-effort)
    try:
        chosen = fitted[best["name"]] if best else None
        if chosen is not None:
            y_pred = chosen.predict(Xte)
            proba = None
            if is_class and is_binary and hasattr(chosen, "predict_proba"):
                try:
                    proba = chosen.predict_proba(Xte)[:, 1]
                except Exception:
                    proba = None

            def _to_py(v):
                try:
                    if isinstance(v, np.generic):
                        return v.item()
                except Exception:
                    pass
                return v

            idx_list = [_to_py(i) for i in Xte.index.tolist()]
            result["pred_test"] = {
                "index": idx_list,
                "y_true": [_to_py(v) for v in yte.tolist()],
                "y_pred": [
                    _to_py(v)
                    for v in (
                        y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
                    )
                ],
            }
            if proba is not None:
                result["pred_test"]["proba"] = [
                    _to_py(v)
                    for v in (
                        proba.tolist() if hasattr(proba, "tolist") else list(proba)
                    )
                ]
    except Exception:
        pass

    result["leaderboard"] = leaderboard
    result["best"] = best
    # Attach selected tools and feature counts for telemetry/result
    result["selected_tools"] = [name for name, _ in cands]
    result["features"] = {"numeric": len(num_cols), "categorical": len(cat_cols)}

    result["task"] = "classification" if is_class else "regression"
    # Minimal modeling.json write for resumability
    try:
        import json

        (JOBS_DIR / job_id / "modeling.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    result["explain"] = explain
    return result
