"""
Deterministic causal effect estimation tools.

Phase 4: Implements ATE estimation methods for binary treatments.
All operations are deterministic (fixed seeds, stable rounding).

Methods:
1. Regression Adjustment ATE (baseline)
2. IPW ATE (Inverse Probability Weighting)

Requirements:
- Binary treatment only
- CausalReadinessReport.readiness_status == PASS
- CausalConfirmations.ok_to_estimate == True
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from packages.contracts.models import CausalEstimateArtifact

# =============================================================================
# Constants
# =============================================================================

RANDOM_SEED = 42
PROPENSITY_TRIM_LOW = 0.01  # Trim propensities below this
PROPENSITY_TRIM_HIGH = 0.99  # Trim propensities above this


class EstimationError(Exception):
    """Raised when estimation fails."""


def _load_dataset(dataset_id: str, datasets_dir: Path) -> pd.DataFrame:
    """Load a dataset by ID from the datasets directory."""
    dataset_path = datasets_dir / dataset_id / "data.csv"
    if not dataset_path.exists():
        raise EstimationError(f"Dataset not found: {dataset_id}")
    return pd.read_csv(dataset_path)


def _load_metadata(dataset_id: str, datasets_dir: Path) -> dict:
    """Load dataset metadata."""
    metadata_path = datasets_dir / dataset_id / "metadata.json"
    if not metadata_path.exists():
        raise EstimationError(f"Metadata not found for dataset: {dataset_id}")
    with open(metadata_path) as f:
        return json.load(f)


def _prepare_data(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for estimation: clean, encode treatment, scale covariates."""
    all_cols = [treatment_col, outcome_col] + covariates
    df_clean = df[all_cols].dropna().copy()

    if len(df_clean) < 20:
        raise EstimationError(f"Insufficient observations: {len(df_clean)}")

    # Get treatment values
    treatment_vals = sorted(df_clean[treatment_col].unique(), key=str)
    if len(treatment_vals) != 2:
        raise EstimationError(f"Treatment must be binary, found {len(treatment_vals)} values")

    # Encode treatment as 0/1
    treatment_map = {treatment_vals[0]: 0, treatment_vals[1]: 1}
    t = df_clean[treatment_col].map(treatment_map).to_numpy().astype(float)
    y = df_clean[outcome_col].to_numpy().astype(float)

    # Prepare covariates
    x_df = df_clean[covariates].copy()
    for col in x_df.columns:
        if not pd.api.types.is_numeric_dtype(x_df[col]):
            x_df[col] = pd.Categorical(x_df[col]).codes.astype(float)

    x = x_df.to_numpy()

    return df_clean, t, y, x


def _compute_hc3_se(y: np.ndarray, y_pred: np.ndarray, x_aug: np.ndarray) -> float:
    """Compute HC3 heteroskedasticity-robust standard error for ATE coefficient."""
    residuals = y - y_pred
    leverage = np.sum(x_aug * np.linalg.lstsq(x_aug.T @ x_aug, x_aug.T, rcond=None)[0].T, axis=1)
    leverage = np.clip(leverage, 0, 0.999)  # Prevent division by zero
    hc3_resid = residuals / (1 - leverage)
    meat = x_aug.T @ np.diag(hc3_resid**2) @ x_aug
    bread = np.linalg.inv(x_aug.T @ x_aug)
    vcov = bread @ meat @ bread
    return float(np.sqrt(vcov[1, 1]))  # SE for treatment coefficient


def regression_adjustment_ate(
    dataset_id: str,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Compute ATE using regression adjustment.

    Model: Y = a + b*T + c*X + e
    ATE = b (coefficient on treatment)

    Returns CausalEstimateArtifact + TableArtifact with regression summary.
    """
    df = _load_dataset(dataset_id, datasets_dir)
    df_clean, t, y, x = _prepare_data(df, treatment_col, outcome_col, covariates)
    n_used = len(df_clean)

    warnings = []

    # Scale covariates for numerical stability
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x) if x.shape[1] > 0 else x

    # Augmented design matrix: [1, T, X_scaled]
    intercept = np.ones((n_used, 1))
    treatment_col_arr = t.reshape(-1, 1)
    if x_scaled.shape[1] > 0:
        x_aug = np.hstack([intercept, treatment_col_arr, x_scaled])
    else:
        x_aug = np.hstack([intercept, treatment_col_arr])

    # Fit OLS: Y ~ T + X
    model = LinearRegression(fit_intercept=False)  # Intercept already in x_aug
    model.fit(x_aug, y)

    ate = float(model.coef_[1])  # Coefficient on treatment
    y_pred = model.predict(x_aug)

    # HC3 robust standard error
    ate_se = _compute_hc3_se(y, y_pred, x_aug)

    # 95% CI (normal approximation)
    z = 1.96
    ci_low = ate - z * ate_se
    ci_high = ate + z * ate_se

    # Round for stability
    ate = round(ate, 4)
    ci_low = round(ci_low, 4)
    ci_high = round(ci_high, 4)

    estimate_artifact = CausalEstimateArtifact(
        method="regression_adjustment",
        estimand="ATE",
        estimate=ate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_used=n_used,
        covariates=covariates,
        warnings=warnings,
    )

    return {"estimate_artifact": estimate_artifact.model_dump()}


def ipw_ate(
    dataset_id: str,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Compute ATE using Inverse Probability Weighting (IPW).

    ATE = E[Y * T / e(X)] - E[Y * (1-T) / (1 - e(X))]

    Propensity scores are trimmed to [0.01, 0.99] for stability.
    Returns CausalEstimateArtifact + TableArtifact with propensity summary.
    """
    df = _load_dataset(dataset_id, datasets_dir)
    df_clean, t, y, x = _prepare_data(df, treatment_col, outcome_col, covariates)
    n_used = len(df_clean)

    warnings = []

    # Scale covariates
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x) if x.shape[1] > 0 else x

    # Fit propensity model (deterministic via random_state)
    ps_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, solver="lbfgs")
    ps_model.fit(x_scaled, t)
    propensity = ps_model.predict_proba(x_scaled)[:, 1]

    # Trim extreme propensities
    n_trimmed = int(np.sum((propensity < PROPENSITY_TRIM_LOW) | (propensity > PROPENSITY_TRIM_HIGH)))
    if n_trimmed > 0:
        warnings.append(f"Trimmed {n_trimmed} observations with extreme propensities.")
    propensity = np.clip(propensity, PROPENSITY_TRIM_LOW, PROPENSITY_TRIM_HIGH)

    # IPW estimator
    weights_treated = t / propensity
    weights_control = (1 - t) / (1 - propensity)

    ate_treated = np.sum(y * weights_treated) / np.sum(weights_treated)
    ate_control = np.sum(y * weights_control) / np.sum(weights_control)
    ate = float(ate_treated - ate_control)

    # Bootstrap CI (deterministic via Generator)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_ates = []
    n_bootstrap = 200
    for _ in range(n_bootstrap):
        idx = rng.choice(n_used, size=n_used, replace=True)
        y_b, t_b, ps_b = y[idx], t[idx], propensity[idx]
        w_t = t_b / ps_b
        w_c = (1 - t_b) / (1 - ps_b)
        ate_t = np.sum(y_b * w_t) / np.sum(w_t)
        ate_c = np.sum(y_b * w_c) / np.sum(w_c)
        bootstrap_ates.append(ate_t - ate_c)

    ci_low = float(np.percentile(bootstrap_ates, 2.5))
    ci_high = float(np.percentile(bootstrap_ates, 97.5))

    # Round for stability
    ate = round(ate, 4)
    ci_low = round(ci_low, 4)
    ci_high = round(ci_high, 4)

    estimate_artifact = CausalEstimateArtifact(
        method="ipw",
        estimand="ATE",
        estimate=ate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_used=n_used,
        covariates=covariates,
        warnings=warnings,
    )

    # Propensity summary table
    ps_summary = {
        "type": "table",
        "headers": ["Statistic", "Value"],
        "rows": [
            ["Mean", f"{np.mean(propensity):.4f}"],
            ["Std", f"{np.std(propensity):.4f}"],
            ["Min", f"{np.min(propensity):.4f}"],
            ["Max", f"{np.max(propensity):.4f}"],
            ["% Treated", f"{np.mean(t) * 100:.1f}%"],
        ],
    }

    return {
        "estimate_artifact": estimate_artifact.model_dump(),
        "propensity_summary": ps_summary,
    }


def select_estimator(
    positivity_status: str,
    n_covariates: int,
) -> str:
    """
    Select the best estimator based on diagnostic results.

    Rules:
    - If positivity PASS and covariates > 0: prefer IPW
    - Otherwise: regression_adjustment (more robust)
    """
    if positivity_status == "PASS" and n_covariates > 0:
        return "ipw"
    return "regression_adjustment"


def run_causal_estimation(
    dataset_id: str,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    datasets_dir: Path,
    method: str = "regression_adjustment",
) -> dict[str, Any]:
    """
    Run causal estimation with the specified method.

    Args:
        dataset_id: Dataset identifier
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        covariates: List of covariate column names
        datasets_dir: Path to datasets directory
        method: Estimation method ("regression_adjustment" or "ipw")

    Returns:
        Dict with estimate_artifact and optional supporting artifacts
    """
    if method == "ipw":
        return ipw_ate(dataset_id, treatment_col, outcome_col, covariates, datasets_dir)
    else:
        return regression_adjustment_ate(
            dataset_id, treatment_col, outcome_col, covariates, datasets_dir
        )

