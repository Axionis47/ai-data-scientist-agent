"""Statistical Testing Service

Provides hypothesis testing and statistical inference capabilities.

Public API:
- run_statistical_tests(df, target, config) -> Dict with comprehensive test results
- test_normality(series) -> Shapiro-Wilk or D'Agostino-Pearson normality test
- test_correlation_significance(x, y) -> Pearson or Spearman correlation with p-value
- test_group_differences(df, group_col, value_col) -> t-test, ANOVA, Mann-Whitney, Kruskal-Wallis
- test_independence(df, col1, col2) -> Chi-square or Fisher's exact with Cramér's V
- test_paired_samples(before, after) -> Paired t-test or Wilcoxon signed-rank with Cohen's d
- test_proportions(s1, n1, s2, n2) -> Z-test for two proportions
- cohens_d(group1, group2) -> Effect size for group comparisons
- eta_squared(df, group_col, value_col) -> Effect size for ANOVA
- benjamini_hochberg_correction(p_values) -> FDR correction for multiple comparisons
- ks_test(series, distribution) -> Kolmogorov-Smirnov goodness of fit

Design:
- All functions return structured dicts with 'test_name', 'statistic', 'p_value', 'interpretation'
- Graceful handling of edge cases (small samples, constant values)
- No side effects - pure functions
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
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
                _, p_value = stats.fisher_exact(contingency.values)
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


def test_paired_samples(
    before: pd.Series, after: pd.Series, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if there is a significant difference between paired samples.

    Uses paired t-test for normal data, Wilcoxon signed-rank otherwise.
    """
    combined = pd.DataFrame({"before": before, "after": after}).dropna()
    if len(combined) < 5:
        return {
            "test_name": "paired_samples",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Insufficient paired data (n={len(combined)})",
        }

    diff = combined["after"] - combined["before"]

    try:
        # Check normality of differences
        norm_result = test_normality(diff)
        is_normal = norm_result.get("is_normal", False)

        if is_normal:
            stat, p_value = stats.ttest_rel(combined["before"], combined["after"])
            test_used = "Paired t-test"
        else:
            stat, p_value = stats.wilcoxon(combined["before"], combined["after"])
            test_used = "Wilcoxon signed-rank"

        is_significant = p_value < alpha
        mean_diff = float(diff.mean())

        # Cohen's d for effect size
        std_diff = diff.std()
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        effect = (
            "large" if abs(cohens_d) > 0.8
            else "medium" if abs(cohens_d) > 0.5
            else "small"
        )

        interpretation = (
            f"{test_used}: {'Significant' if is_significant else 'No significant'} "
            f"difference (p={p_value:.4f}, mean diff={mean_diff:.4f}, Cohen's d={cohens_d:.3f} - {effect})"
        )

        return {
            "test_name": test_used,
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "mean_difference": mean_diff,
            "cohens_d": float(cohens_d),
            "effect_size": effect,
            "n_pairs": len(combined),
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "paired_samples",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Test failed: {e}",
        }


def test_proportions(
    successes1: int, n1: int, successes2: int, n2: int, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if two proportions are significantly different (Z-test for proportions)."""
    if n1 < 10 or n2 < 10:
        return {
            "test_name": "proportions",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": "Sample sizes too small (need n >= 10)",
        }

    try:
        p1 = successes1 / n1
        p2 = successes2 / n2
        p_pooled = (successes1 + successes2) / (n1 + n2)

        # Z-test
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        if se < 1e-10:
            return {
                "test_name": "proportions",
                "statistic": None,
                "p_value": None,
                "is_significant": None,
                "interpretation": "Proportions are identical or near-zero variance",
            }

        z_stat = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

        is_significant = p_value < alpha

        interpretation = (
            f"Z-test for proportions: {'Significant' if is_significant else 'No significant'} "
            f"difference (p1={p1:.3f}, p2={p2:.3f}, z={z_stat:.3f}, p={p_value:.4f})"
        )

        return {
            "test_name": "Z-test for proportions",
            "proportion1": float(p1),
            "proportion2": float(p2),
            "statistic": float(z_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "difference": float(p1 - p2),
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "proportions",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Test failed: {e}",
        }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """Calculate eta-squared effect size for ANOVA."""
    try:
        data = df[[group_col, value_col]].dropna()
        groups = data[group_col].unique()

        if len(groups) < 2:
            return {"eta_squared": None, "interpretation": "Need at least 2 groups"}

        group_data = [data[data[group_col] == g][value_col].values for g in groups]

        # Calculate SS_between and SS_total
        grand_mean = data[value_col].mean()

        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in group_data
        )
        ss_total = ((data[value_col] - grand_mean) ** 2).sum()

        if ss_total < 1e-10:
            return {"eta_squared": 0.0, "interpretation": "No variance in data"}

        eta_sq = ss_between / ss_total

        effect = (
            "large" if eta_sq > 0.14
            else "medium" if eta_sq > 0.06
            else "small"
        )

        return {
            "eta_squared": float(eta_sq),
            "effect_size": effect,
            "interpretation": f"η² = {eta_sq:.4f} ({effect} effect)",
        }
    except Exception as e:
        return {"eta_squared": None, "interpretation": f"Calculation failed: {e}"}


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons."""
    n = len(p_values)
    if n == 0:
        return {"adjusted_p_values": [], "significant": [], "interpretation": "No p-values provided"}

    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])

    # Calculate adjusted p-values
    adjusted = [0.0] * n
    prev_adjusted = 1.0

    for i in range(n - 1, -1, -1):
        orig_idx, p = sorted_pairs[i]
        rank = i + 1
        adjusted_p = min(prev_adjusted, p * n / rank)
        adjusted[orig_idx] = adjusted_p
        prev_adjusted = adjusted_p

    significant = [adj_p < alpha for adj_p in adjusted]

    return {
        "original_p_values": p_values,
        "adjusted_p_values": adjusted,
        "significant": significant,
        "n_significant": sum(significant),
        "interpretation": f"BH correction: {sum(significant)}/{n} tests remain significant at α={alpha}",
    }


def ks_test(series: pd.Series, distribution: str = "norm", alpha: float = 0.05) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test for goodness of fit."""
    arr = series.dropna().values
    n = len(arr)

    if n < 5:
        return {
            "test_name": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "interpretation": f"Insufficient data (n={n})",
        }

    try:
        if distribution == "norm":
            # Standardize data
            standardized = (arr - np.mean(arr)) / np.std(arr)
            stat, p_value = stats.kstest(standardized, "norm")
            dist_name = "normal"
        elif distribution == "expon":
            stat, p_value = stats.kstest(arr, "expon", args=(0, np.mean(arr)))
            dist_name = "exponential"
        elif distribution == "uniform":
            stat, p_value = stats.kstest(arr, "uniform", args=(np.min(arr), np.max(arr) - np.min(arr)))
            dist_name = "uniform"
        else:
            return {"test_name": "Kolmogorov-Smirnov", "interpretation": f"Unknown distribution: {distribution}"}

        is_significant = p_value < alpha
        fits = not is_significant

        interpretation = (
            f"KS test: Data {'fits' if fits else 'does NOT fit'} {dist_name} distribution "
            f"(D={stat:.4f}, p={p_value:.4f})"
        )

        return {
            "test_name": "Kolmogorov-Smirnov",
            "distribution": dist_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "fits_distribution": fits,
            "interpretation": interpretation,
        }
    except Exception as e:
        return {
            "test_name": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
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

