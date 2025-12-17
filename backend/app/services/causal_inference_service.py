"""Causal Inference Service

Provides causal inference and treatment effect estimation capabilities.

Public API:
- estimate_treatment_effect(df, treatment, outcome, confounders) -> Dict
- propensity_score_matching(df, treatment, outcome, covariates) -> Dict
- sensitivity_analysis(effect_result) -> Dict with robustness checks
- detect_causal_question(question) -> Dict with treatment/outcome detection
- run_causal_analysis(df, treatment, outcome, config) -> Dict with full analysis

Design:
- Primary: Uses DoWhy library when available for rigorous causal inference
- Fallback: Simple regression-based estimates when DoWhy unavailable
- Always includes sensitivity analysis and caveats
- Clear documentation of assumptions
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import re

# Optional dependency flags
DOWHY_AVAILABLE = False
ECONML_AVAILABLE = False

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    pass

try:
    from econml.dml import LinearDML
    ECONML_AVAILABLE = True
except ImportError:
    pass


def detect_causal_question(question: str) -> Dict[str, Any]:
    """Detect if a question implies causal inference and extract treatment/outcome.
    
    Looks for patterns like:
    - "What is the effect of X on Y?"
    - "Does X cause Y?"
    - "How does X impact Y?"
    - "What happens to Y if we change X?"
    """
    question_lower = question.lower()
    
    # Causal keywords
    causal_patterns = [
        r"effect\s+of\s+(\w+)\s+on\s+(\w+)",
        r"impact\s+of\s+(\w+)\s+on\s+(\w+)",
        r"does\s+(\w+)\s+cause\s+(\w+)",
        r"how\s+does\s+(\w+)\s+affect\s+(\w+)",
        r"what\s+happens\s+to\s+(\w+)\s+if\s+(?:we\s+)?(?:change|increase|decrease)\s+(\w+)",
        r"causal\s+(?:effect|relationship)\s+(?:of|between)\s+(\w+)\s+(?:and|on)\s+(\w+)",
    ]
    
    causal_keywords = ["effect", "cause", "impact", "causal", "treatment", "intervention"]
    
    is_causal = any(kw in question_lower for kw in causal_keywords)
    
    treatment = None
    outcome = None
    
    for pattern in causal_patterns:
        match = re.search(pattern, question_lower)
        if match:
            is_causal = True
            groups = match.groups()
            if len(groups) >= 2:
                treatment = groups[0]
                outcome = groups[1]
            break
    
    return {
        "is_causal_question": is_causal,
        "treatment": treatment,
        "outcome": outcome,
        "confidence": 0.9 if (treatment and outcome) else (0.6 if is_causal else 0.1),
        "interpretation": (
            f"Detected causal question: effect of '{treatment}' on '{outcome}'"
            if treatment and outcome
            else "Causal intent detected but variables unclear"
            if is_causal
            else "Does not appear to be a causal question"
        ),
    }


def _simple_regression_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Dict[str, Any]:
    """Simple OLS regression-based treatment effect estimate (fallback)."""
    from sklearn.linear_model import LinearRegression
    
    # Prepare data
    all_cols = [treatment] + confounders
    data = df[[outcome] + all_cols].dropna()
    
    if len(data) < 30:
        return {"error": f"Insufficient data (n={len(data)})"}
    
    y = data[outcome].values
    X = data[all_cols].values
    
    # Encode categoricals if needed
    X_encoded = pd.get_dummies(data[all_cols], drop_first=True).values
    
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    # Treatment effect is coefficient of treatment variable
    feature_names = pd.get_dummies(data[all_cols], drop_first=True).columns.tolist()
    treatment_idx = [i for i, name in enumerate(feature_names) if treatment in name]
    
    if not treatment_idx:
        return {"error": f"Treatment variable '{treatment}' not found in features"}
    
    effect = model.coef_[treatment_idx[0]]
    
    # Simple confidence interval (assumes normality)
    residuals = y - model.predict(X_encoded)
    se = np.std(residuals) / np.sqrt(len(y))
    ci_lower = effect - 1.96 * se
    ci_upper = effect + 1.96 * se
    
    return {
        "method": "OLS Regression (fallback)",
        "effect": float(effect),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_observations": len(data),
        "confounders_controlled": confounders,
        "r_squared": float(model.score(X_encoded, y)),
        "caveat": "Simple regression-based estimate. Install 'dowhy' for rigorous causal inference.",
    }


def estimate_treatment_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    method: str = "auto",
) -> Dict[str, Any]:
    """Estimate the causal effect of treatment on outcome.

    Methods:
    - "backdoor": Backdoor adjustment (regression/matching)
    - "iv": Instrumental variables (if instruments available)
    - "propensity": Propensity score weighting
    - "auto": Automatically choose best method
    """
    if treatment not in df.columns:
        return {"error": f"Treatment column '{treatment}' not found"}
    if outcome not in df.columns:
        return {"error": f"Outcome column '{outcome}' not found"}

    # Auto-detect confounders if not provided
    if confounders is None:
        confounders = [
            col for col in df.columns
            if col not in [treatment, outcome]
            and df[col].notna().mean() > 0.5  # Exclude mostly-null columns
        ][:10]  # Limit to 10 confounders

    # Filter to valid confounders
    confounders = [c for c in confounders if c in df.columns]

    result: Dict[str, Any] = {
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
    }

    if DOWHY_AVAILABLE:
        try:
            # Build causal graph
            # Simple graph: confounders -> treatment, confounders -> outcome, treatment -> outcome
            graph = _build_simple_causal_graph(treatment, outcome, confounders)

            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
                graph=graph,
            )

            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # Estimate using backdoor criterion
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
            )

            result.update({
                "method": "DoWhy Backdoor (Linear Regression)",
                "effect": float(estimate.value),
                "estimand": str(identified_estimand),
                "interpretation": estimate.interpret(),
            })

            # Refutation tests
            refutations = []
            try:
                # Random common cause refutation
                refute_random = model.refute_estimate(
                    identified_estimand, estimate,
                    method_name="random_common_cause",
                )
                refutations.append({
                    "test": "random_common_cause",
                    "new_effect": float(refute_random.new_effect),
                    "passed": abs(refute_random.new_effect - estimate.value) < 0.1 * abs(estimate.value),
                })
            except Exception:
                pass

            try:
                # Placebo treatment refutation
                refute_placebo = model.refute_estimate(
                    identified_estimand, estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                )
                refutations.append({
                    "test": "placebo_treatment",
                    "new_effect": float(refute_placebo.new_effect),
                    "passed": abs(refute_placebo.new_effect) < 0.1 * abs(estimate.value),
                })
            except Exception:
                pass

            result["refutations"] = refutations
            result["robust"] = all(r.get("passed", False) for r in refutations) if refutations else None

        except Exception as e:
            # Fallback to simple regression
            result = _simple_regression_effect(df, treatment, outcome, confounders)
            result["dowhy_error"] = str(e)
    else:
        # Fallback to simple regression
        result = _simple_regression_effect(df, treatment, outcome, confounders)

    return result


def _build_simple_causal_graph(
    treatment: str, outcome: str, confounders: List[str]
) -> str:
    """Build a simple DOT graph string for DoWhy."""
    edges = []
    for c in confounders:
        edges.append(f'"{c}" -> "{treatment}"')
        edges.append(f'"{c}" -> "{outcome}"')
    edges.append(f'"{treatment}" -> "{outcome}"')

    graph = "digraph {\n  " + ";\n  ".join(edges) + ";\n}"
    return graph


def propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    n_neighbors: int = 1,
) -> Dict[str, Any]:
    """Estimate treatment effect using propensity score matching."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    # Prepare data
    data = df[[treatment, outcome] + covariates].dropna()

    if len(data) < 50:
        return {"error": f"Insufficient data for matching (n={len(data)})"}

    # Check treatment is binary
    if data[treatment].nunique() != 2:
        return {"error": "Treatment must be binary for propensity score matching"}

    # Encode covariates
    X = pd.get_dummies(data[covariates], drop_first=True)
    T = data[treatment].values
    Y = data[outcome].values

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    # Match treated to control
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    if len(treated_idx) < 10 or len(control_idx) < 10:
        return {"error": "Too few treated or control units"}

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))

    # Calculate ATT (Average Treatment Effect on Treated)
    matched_control_outcomes = Y[control_idx][indices].mean(axis=1)
    treated_outcomes = Y[treated_idx]

    att = float(np.mean(treated_outcomes - matched_control_outcomes))
    att_se = float(np.std(treated_outcomes - matched_control_outcomes) / np.sqrt(len(treated_outcomes)))

    return {
        "method": "Propensity Score Matching",
        "effect": att,
        "effect_type": "ATT (Average Treatment Effect on Treated)",
        "se": att_se,
        "ci_lower": att - 1.96 * att_se,
        "ci_upper": att + 1.96 * att_se,
        "n_treated": len(treated_idx),
        "n_control": len(control_idx),
        "n_matched": len(treated_idx) * n_neighbors,
        "mean_propensity_treated": float(propensity_scores[treated_idx].mean()),
        "mean_propensity_control": float(propensity_scores[control_idx].mean()),
        "interpretation": f"Average effect of treatment on treated: {att:.4f} (95% CI: [{att-1.96*att_se:.4f}, {att+1.96*att_se:.4f}])",
    }


def run_causal_analysis(
    df: pd.DataFrame,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    confounders: Optional[List[str]] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run comprehensive causal analysis.

    Returns:
    - effect_estimate: Primary treatment effect estimate
    - matching_estimate: Propensity score matching estimate
    - assumptions: List of causal assumptions and their validity
    - sensitivity: Sensitivity analysis results
    - recommendations: Suggested next steps
    """
    config = config or {}
    results: Dict[str, Any] = {
        "treatment": treatment,
        "outcome": outcome,
        "effect_estimate": {},
        "matching_estimate": {},
        "assumptions": [],
        "recommendations": [],
        "summary": {},
    }

    if not treatment or not outcome:
        results["error"] = "Treatment and outcome must be specified"
        return results

    if treatment not in df.columns or outcome not in df.columns:
        results["error"] = f"Treatment '{treatment}' or outcome '{outcome}' not found in data"
        return results

    # Auto-detect confounders
    if confounders is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        confounders = [
            c for c in (numeric_cols + cat_cols)
            if c not in [treatment, outcome]
            and df[c].notna().mean() > 0.5
            and df[c].nunique() < 100
        ][:10]

    results["confounders"] = confounders

    # 1. Primary effect estimate
    results["effect_estimate"] = estimate_treatment_effect(
        df, treatment, outcome, confounders
    )

    # 2. Propensity score matching (if treatment is binary)
    if df[treatment].nunique() == 2:
        numeric_confounders = [c for c in confounders if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric_confounders) >= 2:
            results["matching_estimate"] = propensity_score_matching(
                df, treatment, outcome, numeric_confounders
            )

    # 3. Document assumptions
    results["assumptions"] = [
        {
            "name": "No unmeasured confounding",
            "description": "All variables that affect both treatment and outcome are observed",
            "testable": False,
            "recommendation": "Consider sensitivity analysis or find instrumental variables",
        },
        {
            "name": "Positivity",
            "description": "All units have non-zero probability of receiving treatment",
            "testable": True,
            "status": _check_positivity(df, treatment, confounders),
        },
        {
            "name": "SUTVA",
            "description": "Treatment of one unit doesn't affect outcomes of others",
            "testable": False,
            "recommendation": "Consider if interference between units is possible",
        },
    ]

    # 4. Recommendations based on results
    effect = results["effect_estimate"].get("effect")
    ci_lower = results["effect_estimate"].get("ci_lower")
    ci_upper = results["effect_estimate"].get("ci_upper")

    if effect is not None:
        if ci_lower is not None and ci_upper is not None:
            if ci_lower * ci_upper > 0:  # CI doesn't cross zero
                results["recommendations"].append(
                    f"Effect is statistically significant ({effect:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )
            else:
                results["recommendations"].append(
                    "Effect is NOT statistically significant (CI includes zero)"
                )

        # Effect size interpretation
        if abs(effect) < 0.1:
            results["recommendations"].append("Effect size is small")
        elif abs(effect) < 0.5:
            results["recommendations"].append("Effect size is moderate")
        else:
            results["recommendations"].append("Effect size is large")

    if not DOWHY_AVAILABLE:
        results["recommendations"].append(
            "Install 'dowhy' package for more rigorous causal inference with refutation tests"
        )

    # Summary
    results["summary"] = {
        "treatment": treatment,
        "outcome": outcome,
        "n_confounders": len(confounders),
        "estimated_effect": effect,
        "method_used": results["effect_estimate"].get("method"),
        "is_robust": results["effect_estimate"].get("robust"),
        "dowhy_available": DOWHY_AVAILABLE,
    }

    return results


def _check_positivity(
    df: pd.DataFrame, treatment: str, confounders: List[str]
) -> str:
    """Check if positivity assumption is likely satisfied."""
    try:
        # For binary treatment, check if treated/control exist in all strata
        if df[treatment].nunique() != 2:
            return "Not applicable (treatment is not binary)"

        # Simple check: is there overlap in propensity scores?
        from sklearn.linear_model import LogisticRegression

        numeric_conf = [c for c in confounders if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric_conf) < 1:
            return "Cannot assess (no numeric confounders)"

        data = df[[treatment] + numeric_conf].dropna()
        X = data[numeric_conf].values
        T = data[treatment].values

        model = LogisticRegression(max_iter=500)
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]

        # Check overlap
        ps_treated = ps[T == 1]
        ps_control = ps[T == 0]

        overlap = (
            ps_control.max() > ps_treated.min() and
            ps_treated.max() > ps_control.min()
        )

        if overlap:
            return "Likely satisfied (propensity score overlap exists)"
        else:
            return "Violated (no propensity score overlap)"
    except Exception as e:
        return f"Cannot assess: {e}"

