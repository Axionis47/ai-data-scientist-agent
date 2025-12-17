"""Statistical Testing Service

Provides hypothesis testing and statistical inference capabilities.

Public API:
- run_statistical_tests(df, target, config) -> Dict with test results
- test_normality(series) -> Dict with test name, statistic, p-value, interpretation
- test_correlation_significance(x, y) -> Dict with correlation, p-value
- test_group_differences(df, group_col, value_col) -> Dict (t-test or ANOVA)
- test_independence(df, col1, col2) -> Dict (chi-square for categoricals)

Design:
- All functions return structured dicts with 'test_name', 'statistic', 'p_value', 'interpretation'
- Graceful handling of edge cases (small samples, constant values)
- No side effects - pure functions
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

# scipy.stats is a standard dependency
from scipy import stats


def test_normality(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """Test if a numeric series follows a normal distribution.
    
    Uses Shapiro-Wilk for n < 5000, otherwise uses D'Agostino-Pearson.
    """
    arr = series.dropna().values
    n = len(arr)
    
    if n < 8:
        return {
            "test_name": "normality",
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "interpretation": f"Insufficient data (n={n}, need ≥8)",
        }
    
    if np.std(arr) < 1e-10:
        return {
            "test_name": "normality",
            "statistic": None,
            "p_value": None,
            "is_normal": False,
            "interpretation": "Constant or near-constant values",
        }
    
    try:
        if n < 5000:
            stat, p_value = stats.shapiro(arr[:5000])  # Shapiro-Wilk
            test_used = "Shapiro-Wilk"
        else:
            stat, p_value = stats.normaltest(arr)  # D'Agostino-Pearson
            test_used = "D'Agostino-Pearson"
        
        is_normal = p_value > alpha
        interpretation = (
            f"{test_used}: Data appears normally distributed (p={p_value:.4f})"
            if is_normal
            else f"{test_used}: Data is NOT normally distributed (p={p_value:.4f})"
        )
        
        return {
            "test_name": test_used,
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": is_normal,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "normality",
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "interpretation": f"Test failed: {e}",
        }


def test_correlation_significance(
    x: pd.Series, y: pd.Series, method: str = "auto", alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if correlation between two numeric series is statistically significant.
    
    Uses Pearson for normal data, Spearman otherwise.
    """
    # Align and drop NAs
    combined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(combined) < 5:
        return {
            "test_name": "correlation",
            "correlation": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Insufficient data (n={len(combined)})",
        }
    
    x_arr = combined["x"].values
    y_arr = combined["y"].values
    
    # Determine method
    if method == "auto":
        # Use Spearman if either series is non-normal
        norm_x = test_normality(pd.Series(x_arr))
        norm_y = test_normality(pd.Series(y_arr))
        use_pearson = norm_x.get("is_normal", False) and norm_y.get("is_normal", False)
    else:
        use_pearson = method.lower() == "pearson"
    
    try:
        if use_pearson:
            corr, p_value = stats.pearsonr(x_arr, y_arr)
            test_used = "Pearson"
        else:
            corr, p_value = stats.spearmanr(x_arr, y_arr)
            test_used = "Spearman"
        
        is_significant = p_value < alpha
        strength = (
            "strong" if abs(corr) > 0.7
            else "moderate" if abs(corr) > 0.4
            else "weak"
        )
        direction = "positive" if corr > 0 else "negative"
        
        interpretation = (
            f"{test_used}: {strength} {direction} correlation (r={corr:.3f}, p={p_value:.4f})"
            + (" - statistically significant" if is_significant else " - NOT significant")
        )
        
        return {
            "test_name": test_used,
            "correlation": float(corr),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "strength": strength,
            "direction": direction,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "correlation",
            "correlation": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Test failed: {e}",
        }


def test_group_differences(
    df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if there are significant differences between groups.

    Uses t-test for 2 groups, ANOVA for 3+ groups.
    Falls back to non-parametric tests if normality assumption is violated.
    """
    data = df[[group_col, value_col]].dropna()
    groups = data[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        return {
            "test_name": "group_difference",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Need at least 2 groups (found {n_groups})",
        }

    group_data = [data[data[group_col] == g][value_col].values for g in groups]

    # Check sample sizes
    min_size = min(len(g) for g in group_data)
    if min_size < 3:
        return {
            "test_name": "group_difference",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": "Groups too small (min size < 3)",
        }

    try:
        # Check normality across all groups
        all_normal = all(
            test_normality(pd.Series(g)).get("is_normal", False)
            for g in group_data if len(g) >= 8
        )

        if n_groups == 2:
            if all_normal:
                # Levene's test for equal variances
                _, levene_p = stats.levene(*group_data)
                equal_var = levene_p > 0.05
                stat, p_value = stats.ttest_ind(*group_data, equal_var=equal_var)
                test_used = "Independent t-test" + (" (Welch)" if not equal_var else "")
            else:
                stat, p_value = stats.mannwhitneyu(*group_data, alternative='two-sided')
                test_used = "Mann-Whitney U"
        else:
            if all_normal:
                stat, p_value = stats.f_oneway(*group_data)
                test_used = "One-way ANOVA"
            else:
                stat, p_value = stats.kruskal(*group_data)
                test_used = "Kruskal-Wallis"

        is_significant = p_value < alpha
        group_means = {str(g): float(np.mean(gd)) for g, gd in zip(groups, group_data)}

        interpretation = (
            f"{test_used}: {'Significant' if is_significant else 'No significant'} "
            f"difference between groups (p={p_value:.4f})"
        )

        return {
            "test_name": test_used,
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "n_groups": n_groups,
            "group_means": group_means,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "group_difference",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Test failed: {e}",
        }


def test_independence(
    df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if two categorical variables are independent (chi-square test)."""
    try:
        contingency = pd.crosstab(df[col1], df[col2])

        # Check minimum expected frequencies
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        min_expected = expected.min()
        if min_expected < 5:
            # Use Fisher's exact test for 2x2, otherwise warn
            if contingency.shape == (2, 2):
                odds, p_value = stats.fisher_exact(contingency.values)
                test_used = "Fisher's exact"
            else:
                test_used = "Chi-square (low expected freq warning)"
        else:
            test_used = "Chi-square"

        is_significant = p_value < alpha

        # Calculate Cramér's V for effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        effect = (
            "strong" if cramers_v > 0.5
            else "moderate" if cramers_v > 0.3
            else "weak"
        )

        interpretation = (
            f"{test_used}: Variables are {'dependent' if is_significant else 'independent'} "
            f"(p={p_value:.4f}, Cramér's V={cramers_v:.3f} - {effect} association)"
        )

        return {
            "test_name": test_used,
            "statistic": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "is_significant": is_significant,
            "cramers_v": float(cramers_v),
            "effect_size": effect,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "independence",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Test failed: {e}",
        }


def run_statistical_tests(
    df: pd.DataFrame, target: Optional[str] = None, config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Run comprehensive statistical tests on the dataset.

    Returns a structured dict with:
    - normality_tests: Normality test for each numeric column
    - correlation_tests: Significant correlations with target
    - group_tests: Group difference tests for categorical vs target
    - independence_tests: Chi-square tests between categoricals
    """
    config = config or {}
    results: Dict[str, Any] = {
        "normality_tests": [],
        "correlation_tests": [],
        "group_tests": [],
        "independence_tests": [],
        "summary": {},
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Normality tests for numeric columns
    for col in numeric_cols[:20]:  # Limit to 20 columns
        if target and col == target:
            continue
        result = test_normality(df[col])
        result["column"] = col
        results["normality_tests"].append(result)

    # 2. Correlation significance tests with target
    if target and target in df.columns:
        target_series = df[target]
        if np.issubdtype(target_series.dtype, np.number):
            for col in numeric_cols[:20]:
                if col == target:
                    continue
                result = test_correlation_significance(df[col], target_series)
                result["feature"] = col
                result["target"] = target
                results["correlation_tests"].append(result)

    # 3. Group difference tests (categorical features vs numeric target)
    if target and target in df.columns:
        target_series = df[target]
        if np.issubdtype(target_series.dtype, np.number):
            for col in cat_cols[:10]:
                if df[col].nunique() > 20:
                    continue  # Skip high-cardinality
                result = test_group_differences(df, col, target)
                result["feature"] = col
                result["target"] = target
                results["group_tests"].append(result)

    # 4. Independence tests between categoricals
    if len(cat_cols) >= 2:
        tested = set()
        for i, col1 in enumerate(cat_cols[:5]):
            for col2 in cat_cols[i+1:6]:
                if (col1, col2) in tested or (col2, col1) in tested:
                    continue
                if df[col1].nunique() > 20 or df[col2].nunique() > 20:
                    continue
                result = test_independence(df, col1, col2)
                result["column1"] = col1
                result["column2"] = col2
                results["independence_tests"].append(result)
                tested.add((col1, col2))

    # Summary
    n_normal = sum(1 for t in results["normality_tests"] if t.get("is_normal"))
    n_sig_corr = sum(1 for t in results["correlation_tests"] if t.get("is_significant"))
    n_sig_groups = sum(1 for t in results["group_tests"] if t.get("is_significant"))
    n_dependent = sum(1 for t in results["independence_tests"] if t.get("is_significant"))

    results["summary"] = {
        "n_numeric_columns": len(numeric_cols),
        "n_categorical_columns": len(cat_cols),
        "n_normal_distributions": n_normal,
        "n_significant_correlations": n_sig_corr,
        "n_significant_group_differences": n_sig_groups,
        "n_dependent_categoricals": n_dependent,
    }

    return results

