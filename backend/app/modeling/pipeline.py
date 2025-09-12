from typing import Dict, Any, Optional
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, root_mean_squared_error, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
import pandas.api.types as ptypes

from ..core.logs import model_decision
from .transformers import TopKCategorical
from .explainers import add_binary_curves, add_regression_diagnostics, add_shap_beeswarm
from ..core.config import (
    JOBS_DIR, BLEND_DELTA, EARLY_STOP_SAMPLE, HGB_MIN_ROWS, CALIBRATE_ENABLED,
    CV_FOLDS, PDP_TOP_NUM, SHAP_ENABLED, SHAP_MAX_ROWS, SEARCH_TIME_BUDGET
)

# Minimal TopK encoder (copied from main to avoid deep deps)
class TopKCategorical:
    def __init__(self, cols=None, k=50):
        self.cols = cols
        self.k = k
        self.keep_: Dict[str, set] = {}
    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X
        cols = self.cols or list(df.columns)
        for c in cols:
            vc = df[c].astype(str).value_counts()
            self.keep_[c] = set(vc.head(self.k).index.tolist())
        return self
    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X
        for c in (self.cols or list(df.columns)):
            df[c] = df[c].astype(str)
            df[c] = df[c].where(df[c].isin(self.keep_.get(c, set())), other="__OTHER__")
        return df


def _slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-","_")) else "-" for ch in (s or "")).strip("-").lower()


def quick_search(name: str, pipe: Pipeline, X, y, is_class: bool) -> Pipeline:
    import time
    if SEARCH_TIME_BUDGET <= 0:
        return pipe
    t0 = time.time()
    try:
        if name.startswith('rf'):
            grid = { 'est__n_estimators': [100, 200], 'est__max_depth': [None, 10, 20] }
        elif name.startswith('hgb'):
            grid = { 'est__max_depth': [None, 6, 12], 'est__learning_rate': [0.05, 0.1] }
        elif name.startswith('logreg'):
            grid = { 'est__C': [0.5, 1.0, 2.0] }
        else:
            grid = {}
        if not grid:
            return pipe
        n_iter = min(4, max(1, sum(len(v) for v in grid.values())))
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iter,
            scoring=('f1' if is_class else 'r2'),
            cv=(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42) if is_class else CV_FOLDS),
            n_jobs=-1, random_state=42
        )
        search.fit(X, y)
        elapsed = time.time() - t0
        model_decision("", f"Quick search {name}: best={getattr(search,'best_score_',None)} in {elapsed:.1f}s")
        return search.best_estimator_
    except Exception as e:
        model_decision("", f"Quick search failed for {name}: {e}")
        return pipe


def run_modeling(job_id: str, df: pd.DataFrame, eda: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # Prepare target and features
    target = manifest.get("target") or eda.get("target")
    if target is None or target not in df.columns:
        raise ValueError("Target not found")
    y = df[target]
    X = df.drop(columns=[target])

    # Columns
    num_cols = [c for c in X.columns if ptypes.is_numeric_dtype(X[c])]
    from pandas.api.types import CategoricalDtype
    cat_cols = [c for c in X.columns if (ptypes.is_object_dtype(X[c]) or isinstance(X[c].dtype, CategoricalDtype))]
    n_rows = len(X)
    cat_ratio = (len(cat_cols) / max(1, len(X.columns)))

    # Preprocess
    topk = TopKCategorical(cols=cat_cols, k=50)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[("topk", topk), ("impute", SimpleImputer(strategy="most_frequent")),
                               ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ], remainder='drop')

    is_class = (not ptypes.is_numeric_dtype(y)) or (y.nunique(dropna=True) <= 10)
    y_unique = int(pd.Series(y).nunique(dropna=True))

    # Feature selection for linear
    use_mi = False
    try:
        Xs = X.head(min(1000, len(X)))
        pre_preview = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[("topk", TopKCategorical(cols=cat_cols, k=50)), ("impute", SimpleImputer(strategy="most_frequent")),
                                   ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
        ], remainder='drop')
        _ = pre_preview.fit_transform(Xs)
        # Guard when cat_cols may be empty or transformer structure differs
        try:
            ohe = getattr(pre_preview, 'transformers_', [])[1][1].named_steps['ohe']
            n_feat = int(ohe.get_feature_names_out().shape[0] + len(num_cols))
        except Exception:
            n_feat = len(num_cols)
        use_mi = n_feat > 100
    except Exception:
        pass

    linear_sel_steps = [("var", VarianceThreshold(0.01))]
    if use_mi:
        if is_class:
            linear_sel_steps.append(("mi", SelectKBest(mutual_info_classif, k=50)))
        else:
            linear_sel_steps.append(("mi", SelectKBest(mutual_info_regression, k=50)))
        model_decision(job_id, "Feature selection: SelectKBest top=50 for linear models")
    else:
        model_decision(job_id, "Feature selection: VarianceThreshold(0.01) only for linear models")

    # Candidates
    if is_class:
        cands = [
            ("logreg", Pipeline([("prep", pre)] + linear_sel_steps + [("est", LogisticRegression(max_iter=200))])),
            ("rf_clf", Pipeline([("prep", pre), ("est", RandomForestClassifier(n_estimators=200, n_jobs=-1))])),
            ("hgb_clf", Pipeline([("prep", pre), ("est", HistGradientBoostingClassifier(random_state=42))]))
        ]
    else:
        cands = [
            ("linreg", Pipeline([("prep", pre)] + linear_sel_steps + [("est", LinearRegression())])),
            ("rf_reg", Pipeline([("prep", pre), ("est", RandomForestRegressor(n_estimators=200, n_jobs=-1))])),
            ("hgb_reg", Pipeline([("prep", pre), ("est", HistGradientBoostingRegressor(random_state=42))]))
        ]

    # Intelligent gating
    gated = []
    for name, pipe in cands:
        if n_rows < HGB_MIN_ROWS and name.startswith("hgb"):
            model_decision(job_id, f"Auto-gating: drop {name} for n_rows<{HGB_MIN_ROWS}")
            continue
        gated.append((name, pipe))
    cands = gated
    if cat_ratio > 0.8:
        cands = sorted(cands, key=lambda kv: (0 if kv[0].startswith("rf") or kv[0].startswith("hgb") else 1))
        model_decision(job_id, f"Prioritizing tree models due to high categorical ratio={cat_ratio:.2f}")

    # Split strategy (simple version here; keep consistent with main if needed)
    from sklearn.model_selection import train_test_split, StratifiedKFold
    strat = y if is_class else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    # Choose safe CV folds for tiny datasets
    if is_class:
        try:
            counts = ytr.value_counts()
            min_class = int(counts.min()) if len(counts) else 0
            eff_folds = max(2, min(CV_FOLDS, min_class)) if min_class >= 2 else 2
        except Exception:
            eff_folds = min(CV_FOLDS, 3)
        try:
            cv_iter = StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=42).split(Xtr, ytr)
        except Exception:
            cv_iter = eff_folds
    else:
        eff_folds = min(CV_FOLDS, max(2, len(Xtr)//5)) if len(Xtr) >= 10 else 2
        cv_iter = eff_folds

    # Early sample evaluation
    sample_eval = None
    if n_rows > 50_000:
        sample_eval = Xtr.sample(n=min(len(Xtr), EARLY_STOP_SAMPLE), random_state=42)
        y_sample = ytr.loc[sample_eval.index]
        model_decision(job_id, f"Large dataset: initial evaluation on sample n={len(sample_eval)}")

    leaderboard = []
    fitted: Dict[str, Any] = {}
    proba_map: Dict[str, Optional[np.ndarray]] = {}
    preds_map: Dict[str, np.ndarray] = {}

    for name, pipe in cands:
        # Optional early sample evaluation
        if sample_eval is not None:
            try:
                pipe.fit(sample_eval, y_sample)
                preds_s = pipe.predict(sample_eval)
                score_s = f1_score(y_sample, preds_s) if is_class else r2_score(y_sample, preds_s)
                if (is_class and score_s < 0.3) or ((not is_class) and score_s < 0.0):
                    model_decision(job_id, f"Early stop: filtering out {name} due to weak sample score={score_s:.3f}")
                    continue
                model_decision(job_id, f"Promising on sample: {name} score={score_s:.3f}, proceeding to CV")
            except Exception as e:
                model_decision(job_id, f"Sample eval failed for {name}: {e}")
        # Quick search (obeys SEARCH_TIME_BUDGET)
        pipe = quick_search(name, pipe, Xtr, ytr, is_class)

        # CV scoring
        if isinstance(cv_iter, int):
            scores = cross_val_score(pipe, Xtr, ytr, scoring=("f1" if is_class else "r2"), cv=cv_iter, n_jobs=-1)
        else:
            scores = []
            for tr_idx, va_idx in cv_iter:
                pipe.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
                preds_cv = pipe.predict(Xtr.iloc[va_idx])
                s = f1_score(ytr.iloc[va_idx], preds_cv) if is_class else r2_score(ytr.iloc[va_idx], preds_cv)
                scores.append(s)
            scores = np.array(scores)
        cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        fitted[name] = pipe
        if is_class:
            if len(pd.Series(y).nunique()) == 2:
                try:
                    proba = pipe.predict_proba(Xte)[:,1]
                    proba_map[name] = proba
                    prec, rec, thr = precision_recall_curve(yte, proba)
                    f1s = 2*prec*rec/(prec+rec+1e-12)
                    t_star = float(thr[f1s[:-1].argmax()]) if len(thr)>0 else 0.5
                    preds_t = (proba >= t_star).astype(int)
                    f1 = f1_score(yte, preds_t)
                except Exception:
                    f1 = f1_score(yte, preds)
            else:
                f1 = f1_score(yte, preds, average="weighted")
            acc = accuracy_score(yte, preds)
            leaderboard.append({"name": name, "f1": float(f1), "acc": float(acc), "cv_mean": cv_mean, "cv_std": cv_std})
        else:
            r2 = r2_score(yte, preds)
            try:
                rmse = root_mean_squared_error(yte, preds)
            except Exception:
                rmse = float(pd.Series(yte - preds).pow(2).mean() ** 0.5)
            leaderboard.append({"name": name, "r2": float(r2), "rmse": float(rmse), "cv_mean": cv_mean, "cv_std": cv_std})

    # Complexity penalty for very small datasets
    if n_rows < 500:
        for m in leaderboard:
            if m['name'].startswith('hgb') or m['name'].startswith('rf'):
                base = (m.get('cv_mean') or (m.get('f1') if is_class else m.get('r2')) or 0.0)
                m['cv_mean'] = float(base) * 0.98
        model_decision(job_id, "Applied complexity penalty to tree/boosting candidates (n_rows<500)")

    # Select best
    if is_class:
        leaderboard.sort(key=lambda m: (m.get("f1") or 0.0), reverse=True)
    else:
        leaderboard.sort(key=lambda m: (m.get("r2") or -1.0), reverse=True)
    best = leaderboard[0] if leaderboard else None

    # Build explain (basic)
    explain: Dict[str, Any] = {}
    try:
        chosen = fitted[best['name']] if best else None
        if chosen is not None:
            subset = min(len(Xte), 2000)
            res = permutation_importance(chosen, Xte.iloc[:subset], yte.iloc[:subset], n_repeats=5, random_state=42)
            explain['importances'] = res.importances_mean.tolist()
            plots_dir = (JOBS_DIR / job_id / "plots"); plots_dir.mkdir(parents=True, exist_ok=True)
            base_url = f"/static/jobs/{job_id}/plots"
            if is_class and hasattr(chosen, "predict_proba") and len(pd.Series(y).nunique()) == 2:
                add_binary_curves(chosen, Xte, yte, plots_dir, base_url, explain)
            if not is_class:
                add_regression_diagnostics(chosen, Xte, yte, plots_dir, base_url, explain)
            # SHAP optional
            if SHAP_ENABLED and len(Xtr) <= SHAP_MAX_ROWS and is_class and hasattr(chosen, "predict_proba"):
                try:
                    add_shap_beeswarm(chosen, Xtr, Xte, plots_dir, base_url, explain)
                except Exception:
                    pass
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
        (JOBS_DIR / job_id / "modeling.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass

    result["explain"] = explain
    return result

