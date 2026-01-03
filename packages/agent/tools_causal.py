"""
Deterministic causal diagnostic tools.

Each tool returns DiagnosticArtifact and/or TableArtifact with PASS/WARN/FAIL status.
All operations are deterministic (fixed seeds, stable sorting, stable rounding).
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Thresholds (documented for transparency)
MISSING_WARN_THRESHOLD = 0.05  # >5% missing = WARN
MISSING_FAIL_THRESHOLD = 0.20  # >20% missing = FAIL
POSITIVITY_WARN_THRESHOLD = 0.05  # Propensity <5% or >95% = WARN
POSITIVITY_FAIL_THRESHOLD = 0.01  # Propensity <1% or >99% = FAIL
SMD_WARN_THRESHOLD = 0.10  # SMD >0.1 = WARN
SMD_FAIL_THRESHOLD = 0.25  # SMD >0.25 = FAIL


class CausalToolError(Exception):
    """Raised when a causal tool encounters an error."""


def _load_dataset(dataset_id: str, datasets_dir: Path) -> pd.DataFrame:
    """Load a dataset by ID from the datasets directory."""
    dataset_path = datasets_dir / dataset_id / "data.csv"
    if not dataset_path.exists():
        raise CausalToolError(f"Dataset not found: {dataset_id}")
    return pd.read_csv(dataset_path)


def _load_metadata(dataset_id: str, datasets_dir: Path) -> dict:
    """Load dataset metadata."""
    metadata_path = datasets_dir / dataset_id / "metadata.json"
    if not metadata_path.exists():
        raise CausalToolError(f"Metadata not found for dataset: {dataset_id}")
    with open(metadata_path) as f:
        return json.load(f)


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Validate columns exist, return missing ones."""
    return [c for c in columns if c not in df.columns]


def infer_candidate_columns(
    question: str,
    column_names: list[str],
    inferred_types: dict[str, str],
) -> dict[str, Any]:
    """
    Infer treatment/outcome candidates from question text and column metadata.
    
    Returns dict with treatment_candidates, outcome_candidates, and confidence.
    """
    question_lower = question.lower()
    
    # Keywords suggesting treatment
    treatment_keywords = [
        "effect of", "impact of", "does", "did", "whether",
        "treatment", "intervention", "exposure", "receiving",
        "assigned", "given", "policy", "program", "experiment",
    ]
    
    # Keywords suggesting outcome
    outcome_keywords = [
        "on", "affect", "influence", "change", "outcome",
        "result", "performance", "success", "failure", "rate",
        "conversion", "retention", "churn", "revenue", "sales",
    ]
    
    treatment_candidates = []
    outcome_candidates = []
    
    # Look for exact column name matches in question
    for col in column_names:
        col_lower = col.lower()
        if col_lower in question_lower or col.replace("_", " ").lower() in question_lower:
            # Determine if treatment or outcome based on context
            col_idx = question_lower.find(col_lower)
            if col_idx == -1:
                col_idx = question_lower.find(col.replace("_", " ").lower())
            
            before_text = question_lower[:col_idx] if col_idx > 0 else ""
            # after_text could be used for more advanced parsing in future
            _ = question_lower[col_idx:] if col_idx >= 0 else ""
            
            is_treatment = any(kw in before_text[-50:] for kw in treatment_keywords)
            is_outcome = any(kw in before_text[-30:] for kw in outcome_keywords)
            
            if is_treatment:
                treatment_candidates.append(col)
            elif is_outcome:
                outcome_candidates.append(col)
            else:
                # Default: binary/categorical -> treatment, numeric -> outcome
                col_type = inferred_types.get(col, "unknown")
                if col_type in ["bool", "category", "object"]:
                    treatment_candidates.append(col)
                elif col_type in ["int", "float"]:
                    outcome_candidates.append(col)
    
    # If no matches, look for common patterns
    if not treatment_candidates:
        for col in column_names:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["treatment", "treated", "group", "arm", "condition", "exposed"]):
                treatment_candidates.append(col)
    
    if not outcome_candidates:
        for col in column_names:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["outcome", "result", "target", "response", "y_", "label"]):
                outcome_candidates.append(col)
    
    return {
        "treatment_candidates": list(set(treatment_candidates)),
        "outcome_candidates": list(set(outcome_candidates)),
        "confidence": "high" if treatment_candidates and outcome_candidates else "low",
    }


def check_treatment_type(
    dataset_id: str,
    treatment_col: str,
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Check if treatment is binary (required for most causal methods).
    
    Returns DiagnosticArtifact with PASS (binary), WARN (multi-class), or FAIL (continuous).
    """
    df = _load_dataset(dataset_id, datasets_dir)
    missing = _validate_columns(df, [treatment_col])
    if missing:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "treatment_type_check",
                "status": "FAIL",
                "details": {"error": f"Column not found: {treatment_col}"},
                "recommendations": [f"Specify a valid treatment column. Available: {list(df.columns)}"],
            }
        }
    
    unique_values = df[treatment_col].dropna().unique()
    n_unique = len(unique_values)
    
    if n_unique == 2:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "treatment_type_check",
                "status": "PASS",
                "details": {
                    "treatment_col": treatment_col,
                    "n_unique": n_unique,
                    "values": [str(v) for v in sorted(unique_values)],
                    "is_binary": True,
                },
                "recommendations": [],
            }
        }
    elif 2 < n_unique <= 10:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "treatment_type_check",
                "status": "WARN",
                "details": {
                    "treatment_col": treatment_col,
                    "n_unique": n_unique,
                    "values": [str(v) for v in sorted(unique_values)[:10]],
                    "is_binary": False,
                },
                "recommendations": [
                    f"Treatment '{treatment_col}' has {n_unique} values. Consider binarizing.",
                    "Specify which value represents 'treated' vs 'control'.",
                ],
            }
        }
    else:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "treatment_type_check",
                "status": "FAIL",
                "details": {
                    "treatment_col": treatment_col,
                    "n_unique": n_unique,
                    "is_binary": False,
                    "is_continuous": True,
                },
                "recommendations": [
                    f"Treatment '{treatment_col}' appears continuous ({n_unique} unique values).",
                    "Define a binary treatment indicator or specify dose-response analysis.",
                ],
            }
        }


def check_missingness(
    dataset_id: str,
    columns: list[str],
    treatment_col: str | None,
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Check missingness in key columns, optionally stratified by treatment.

    Returns TableArtifact + DiagnosticArtifact.
    """
    df = _load_dataset(dataset_id, datasets_dir)
    missing = _validate_columns(df, columns)
    if missing:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "missingness_check",
                "status": "FAIL",
                "details": {"missing_columns": missing},
                "recommendations": [f"Columns not found: {missing}"],
            }
        }

    n_total = len(df)
    results = []
    max_missing_pct = 0.0

    for col in sorted(columns):
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / n_total if n_total > 0 else 0
        max_missing_pct = max(max_missing_pct, pct_missing)

        row = [col, str(n_missing), f"{pct_missing*100:.2f}%"]

        # Check treatment-dependent missingness if treatment specified
        if treatment_col and treatment_col in df.columns:
            treatment_vals = df[treatment_col].dropna().unique()
            if len(treatment_vals) == 2:
                for val in sorted(treatment_vals, key=str):
                    subset = df[df[treatment_col] == val]
                    subset_missing = subset[col].isna().sum() / len(subset) if len(subset) > 0 else 0
                    row.append(f"{subset_missing*100:.2f}%")

        results.append(row)

    # Determine status
    if max_missing_pct > MISSING_FAIL_THRESHOLD:
        status = "FAIL"
        recommendations = [
            f"High missingness detected (>{MISSING_FAIL_THRESHOLD*100:.0f}%).",
            "Consider imputation strategy or restricting analysis to complete cases.",
        ]
    elif max_missing_pct > MISSING_WARN_THRESHOLD:
        status = "WARN"
        recommendations = [
            f"Moderate missingness detected (>{MISSING_WARN_THRESHOLD*100:.0f}%).",
            "Assess whether missingness is random (MCAR/MAR) or systematic (MNAR).",
        ]
    else:
        status = "PASS"
        recommendations = []

    # Build headers
    headers = ["Column", "N Missing", "% Missing"]
    if treatment_col and treatment_col in df.columns:
        treatment_vals = df[treatment_col].dropna().unique()
        if len(treatment_vals) == 2:
            for val in sorted(treatment_vals, key=str):
                headers.append(f"% Missing ({treatment_col}={val})")

    return {
        "table_artifact": {"type": "table", "headers": headers, "rows": results},
        "diagnostic": {
            "type": "diagnostic",
            "name": "missingness_check",
            "status": status,
            "details": {"max_missing_pct": round(max_missing_pct, 4), "columns_checked": columns},
            "recommendations": recommendations,
        },
    }


def check_time_ordering(
    dataset_id: str,
    time_col: str | None,
    treatment_col: str,
    outcome_col: str,
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Check if time ordering can be established (treatment before outcome).

    Returns DiagnosticArtifact with PASS/WARN/FAIL.
    """
    df = _load_dataset(dataset_id, datasets_dir)

    # Check if columns exist
    cols_to_check = [treatment_col, outcome_col]
    if time_col:
        cols_to_check.append(time_col)

    missing = _validate_columns(df, cols_to_check)
    if missing:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "time_ordering_check",
                "status": "FAIL",
                "details": {"missing_columns": missing},
                "recommendations": [f"Columns not found: {missing}"],
            }
        }

    if not time_col:
        # No time column - cannot verify ordering
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "time_ordering_check",
                "status": "WARN",
                "details": {
                    "has_time_col": False,
                    "message": "No time column specified. Cannot verify treatment precedes outcome.",
                },
                "recommendations": [
                    "Specify a time column to verify treatment timing.",
                    "Confirm treatment decision was made before outcome was observed.",
                    "What is the decision/treatment time and outcome measurement horizon?",
                ],
            }
        }

    # Try to parse time column
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        has_valid_time = True
    except Exception:
        has_valid_time = False

    if not has_valid_time:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "time_ordering_check",
                "status": "WARN",
                "details": {
                    "has_time_col": True,
                    "time_col": time_col,
                    "parseable": False,
                },
                "recommendations": [
                    f"Time column '{time_col}' could not be parsed as datetime.",
                    "Confirm the temporal relationship between treatment and outcome.",
                ],
            }
        }

    # Time column exists and is parseable
    return {
        "diagnostic": {
            "type": "diagnostic",
            "name": "time_ordering_check",
            "status": "PASS",
            "details": {
                "has_time_col": True,
                "time_col": time_col,
                "parseable": True,
                "date_range": {
                    "min": df[time_col].min().isoformat(),
                    "max": df[time_col].max().isoformat(),
                },
            },
            "recommendations": [],
        }
    }


def check_leakage(
    dataset_id: str,  # noqa: ARG001
    treatment_col: str,
    outcome_col: str,
    candidate_confounders: list[str],
    datasets_dir: Path,  # noqa: ARG001
) -> dict[str, Any]:
    """
    Check for potential post-treatment leakage in candidate confounders.

    Heuristic: flag columns with suspicious names (after, post, outcome, result, etc.)
    Note: This is a heuristic check based on column names. Full leakage detection
    would require domain knowledge.
    """
    # Suspicious patterns indicating post-treatment variables
    leakage_patterns = [
        r"after", r"post", r"outcome", r"result", r"final",
        r"_y$", r"_out", r"response", r"target", r"label",
    ]

    suspicious_cols = []
    safe_cols = []

    for col in candidate_confounders:
        col_lower = col.lower()
        is_suspicious = any(re.search(pattern, col_lower) for pattern in leakage_patterns)

        if is_suspicious:
            suspicious_cols.append(col)
        else:
            safe_cols.append(col)

    if suspicious_cols:
        status = "WARN"
        recommendations = [
            f"Potentially post-treatment variables detected: {suspicious_cols}",
            "Confounders must be measured BEFORE treatment assignment.",
            "Review these variables and remove if they are affected by treatment.",
        ]
    else:
        status = "PASS"
        recommendations = []

    return {
        "diagnostic": {
            "type": "diagnostic",
            "name": "leakage_check",
            "status": status,
            "details": {
                "suspicious_columns": suspicious_cols,
                "safe_columns": safe_cols,
                "treatment_col": treatment_col,
                "outcome_col": outcome_col,
            },
            "recommendations": recommendations,
        }
    }


def check_positivity_overlap(
    dataset_id: str,
    treatment_col: str,
    confounders: list[str],
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Check positivity/overlap assumption using propensity score distribution.

    For binary treatment: fits logistic regression and checks for extreme propensities.
    Returns TableArtifact with propensity distribution + DiagnosticArtifact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = _load_dataset(dataset_id, datasets_dir)

    # Validate columns
    all_cols = [treatment_col] + confounders
    missing = _validate_columns(df, all_cols)
    if missing:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "positivity_check",
                "status": "FAIL",
                "details": {"missing_columns": missing},
                "recommendations": [f"Columns not found: {missing}"],
            }
        }

    # Check treatment is binary
    unique_vals = df[treatment_col].dropna().unique()
    if len(unique_vals) != 2:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "positivity_check",
                "status": "FAIL",
                "details": {
                    "treatment_col": treatment_col,
                    "n_unique": len(unique_vals),
                    "message": "Positivity check requires binary treatment.",
                },
                "recommendations": ["Binarize treatment before checking positivity."],
            }
        }

    # Prepare data
    df_clean = df[all_cols].dropna()
    if len(df_clean) < 50:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "positivity_check",
                "status": "WARN",
                "details": {"n_complete_cases": len(df_clean)},
                "recommendations": ["Insufficient complete cases for propensity modeling."],
            }
        }

    # Encode treatment as 0/1
    treatment_map = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
    y = df_clean[treatment_col].map(treatment_map)

    # Prepare confounders (numeric only for simplicity)
    X = df_clean[confounders].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.Categorical(X[col]).codes

    # Fit propensity model (deterministic with fixed seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y)

    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    # Compute distribution summary
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pctl_values = np.percentile(propensity_scores, percentiles)

    # Check for violations
    pct_below_fail = (propensity_scores < POSITIVITY_FAIL_THRESHOLD).mean()
    pct_above_fail = (propensity_scores > (1 - POSITIVITY_FAIL_THRESHOLD)).mean()
    pct_below_warn = (propensity_scores < POSITIVITY_WARN_THRESHOLD).mean()
    pct_above_warn = (propensity_scores > (1 - POSITIVITY_WARN_THRESHOLD)).mean()

    extreme_pct = pct_below_fail + pct_above_fail
    borderline_pct = pct_below_warn + pct_above_warn

    if extreme_pct > 0.10:
        status = "FAIL"
        recommendations = [
            f"{extreme_pct*100:.1f}% of units have extreme propensities (<{POSITIVITY_FAIL_THRESHOLD} or >{1-POSITIVITY_FAIL_THRESHOLD}).",
            "Positivity violation detected. Consider trimming extreme propensities or restricting population.",
        ]
    elif borderline_pct > 0.20:
        status = "WARN"
        recommendations = [
            f"{borderline_pct*100:.1f}% of units have borderline propensities.",
            "Overlap may be limited. Consider alternative estimators (e.g., matching, bounds).",
        ]
    else:
        status = "PASS"
        recommendations = []

    # Build table
    headers = ["Percentile", "Propensity Score"]
    rows = [[f"{p}%", f"{v:.4f}"] for p, v in zip(percentiles, pctl_values)]

    return {
        "table_artifact": {"type": "table", "headers": headers, "rows": rows},
        "diagnostic": {
            "type": "diagnostic",
            "name": "positivity_check",
            "status": status,
            "details": {
                "n_observations": len(df_clean),
                "propensity_mean": round(float(propensity_scores.mean()), 4),
                "propensity_std": round(float(propensity_scores.std()), 4),
                "pct_extreme": round(float(extreme_pct), 4),
                "confounders_used": confounders,
            },
            "recommendations": recommendations,
        },
    }


def balance_check_smd(
    dataset_id: str,
    treatment_col: str,
    confounders: list[str],
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Compute Standardized Mean Differences (SMD) for covariates by treatment.

    SMD = (mean_treated - mean_control) / pooled_std
    Returns TableArtifact with SMD values + DiagnosticArtifact.
    """
    df = _load_dataset(dataset_id, datasets_dir)

    # Validate columns
    all_cols = [treatment_col] + confounders
    missing = _validate_columns(df, all_cols)
    if missing:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "balance_check",
                "status": "FAIL",
                "details": {"missing_columns": missing},
                "recommendations": [f"Columns not found: {missing}"],
            }
        }

    # Check treatment is binary
    unique_vals = df[treatment_col].dropna().unique()
    if len(unique_vals) != 2:
        return {
            "diagnostic": {
                "type": "diagnostic",
                "name": "balance_check",
                "status": "FAIL",
                "details": {
                    "treatment_col": treatment_col,
                    "n_unique": len(unique_vals),
                },
                "recommendations": ["Balance check requires binary treatment."],
            }
        }

    # Split by treatment
    sorted_vals = sorted(unique_vals, key=str)
    control_mask = df[treatment_col] == sorted_vals[0]
    treated_mask = df[treatment_col] == sorted_vals[1]

    results = []
    max_smd = 0.0

    for col in sorted(confounders):
        series = df[col]

        # Convert categorical to numeric for SMD
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.Categorical(series).codes
            series = series.astype(float)
            series[series < 0] = np.nan  # Restore NaN for missing

        control_vals = series[control_mask].dropna()
        treated_vals = series[treated_mask].dropna()

        if len(control_vals) < 2 or len(treated_vals) < 2:
            smd = np.nan
            smd_str = "N/A"
        else:
            mean_c = control_vals.mean()
            mean_t = treated_vals.mean()
            var_c = control_vals.var()
            var_t = treated_vals.var()

            # Pooled standard deviation
            pooled_std = np.sqrt((var_c + var_t) / 2)

            if pooled_std > 0:
                smd = abs(mean_t - mean_c) / pooled_std
                smd_str = f"{smd:.4f}"
                max_smd = max(max_smd, smd)
            else:
                smd = 0.0
                smd_str = "0.0000"

        # Determine status for this variable
        if pd.isna(smd):
            var_status = "N/A"
        elif smd > SMD_FAIL_THRESHOLD:
            var_status = "FAIL"
        elif smd > SMD_WARN_THRESHOLD:
            var_status = "WARN"
        else:
            var_status = "PASS"

        results.append([col, smd_str, var_status])

    # Overall status
    if max_smd > SMD_FAIL_THRESHOLD:
        status = "FAIL"
        recommendations = [
            f"Large imbalance detected (SMD > {SMD_FAIL_THRESHOLD}).",
            "Consider propensity score weighting, matching, or regression adjustment.",
        ]
    elif max_smd > SMD_WARN_THRESHOLD:
        status = "WARN"
        recommendations = [
            f"Moderate imbalance detected (SMD > {SMD_WARN_THRESHOLD}).",
            "Adjustment methods recommended to reduce bias.",
        ]
    else:
        status = "PASS"
        recommendations = []

    headers = ["Covariate", "SMD", "Status"]

    return {
        "table_artifact": {"type": "table", "headers": headers, "rows": results},
        "diagnostic": {
            "type": "diagnostic",
            "name": "balance_check",
            "status": status,
            "details": {
                "max_smd": round(float(max_smd), 4) if not pd.isna(max_smd) else None,
                "n_covariates": len(confounders),
                "treatment_values": [str(v) for v in sorted_vals],
            },
            "recommendations": recommendations,
        },
    }

