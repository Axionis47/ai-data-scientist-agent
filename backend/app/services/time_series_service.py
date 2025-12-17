"""Time Series Analysis Service

Provides time series analysis, decomposition, and forecasting capabilities.

Public API:
- detect_time_series(df, config) -> Dict with time column detection
- test_stationarity(series) -> Dict with ADF/KPSS test results
- decompose_time_series(series, freq) -> Dict with trend, seasonal, residual
- generate_lag_features(df, cols, lags) -> DataFrame with lag features
- run_time_series_analysis(df, time_col, target) -> Dict with full analysis
- forecast_arima(series, horizon) -> Dict with forecasts

Design:
- Graceful fallbacks when statsmodels/prophet not available
- Auto-detection of time columns and frequency
- Handles missing time points
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

# Optional dependency flags
STATSMODELS_AVAILABLE = False
PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass


def detect_time_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect potential time/date columns in the dataframe."""
    time_cols = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check column name patterns
        time_patterns = ["date", "time", "timestamp", "created", "updated", "dt", "year", "month", "day"]
        name_match = any(p in col_lower for p in time_patterns)
        
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append({
                "column": col,
                "dtype": "datetime",
                "confidence": 0.95,
                "sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else None,
            })
            continue
        
        # Try parsing as datetime
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
            
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            valid_ratio = parsed.notna().mean()
            if valid_ratio > 0.8:
                time_cols.append({
                    "column": col,
                    "dtype": "parseable",
                    "confidence": float(valid_ratio * 0.9),
                    "sample": str(sample.iloc[0]),
                })
        except Exception:
            if name_match:
                # Might be encoded time (year, month integers)
                if df[col].dtype in [np.int64, np.int32, np.float64]:
                    time_cols.append({
                        "column": col,
                        "dtype": "numeric_time",
                        "confidence": 0.5,
                        "sample": str(sample.iloc[0]),
                    })
    
    # Sort by confidence
    time_cols = sorted(time_cols, key=lambda x: x["confidence"], reverse=True)
    
    return {
        "time_columns": time_cols,
        "best_candidate": time_cols[0]["column"] if time_cols else None,
        "n_candidates": len(time_cols),
    }


def detect_frequency(series: pd.Series) -> Dict[str, Any]:
    """Detect the frequency of a datetime series."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series, errors="coerce")
        except Exception:
            return {"frequency": None, "interpretation": "Cannot parse as datetime"}
    
    series = series.dropna().sort_values()
    if len(series) < 3:
        return {"frequency": None, "interpretation": "Insufficient data"}
    
    # Calculate differences
    diffs = series.diff().dropna()
    median_diff = diffs.median()
    
    # Map to common frequencies
    if median_diff <= pd.Timedelta(seconds=1):
        freq = "S"
        interp = "Seconds"
    elif median_diff <= pd.Timedelta(minutes=1):
        freq = "min"
        interp = "Minutes"
    elif median_diff <= pd.Timedelta(hours=1):
        freq = "h"
        interp = "Hourly"
    elif median_diff <= pd.Timedelta(days=1):
        freq = "D"
        interp = "Daily"
    elif median_diff <= pd.Timedelta(days=7):
        freq = "W"
        interp = "Weekly"
    elif median_diff <= pd.Timedelta(days=31):
        freq = "ME"
        interp = "Monthly"
    elif median_diff <= pd.Timedelta(days=92):
        freq = "QE"
        interp = "Quarterly"
    else:
        freq = "YE"
        interp = "Yearly"
    
    return {
        "frequency": freq,
        "interpretation": interp,
        "median_gap": str(median_diff),
        "n_observations": len(series),
    }


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """Test if a time series is stationary using ADF and KPSS tests.

    Returns combined interpretation:
    - Both agree stationary -> "stationary"
    - Both agree non-stationary -> "non_stationary"
    - Disagree -> "trend_stationary" or "difference_stationary"
    """
    if not STATSMODELS_AVAILABLE:
        return {
            "adf": None,
            "kpss": None,
            "is_stationary": None,
            "interpretation": "statsmodels not installed",
        }

    arr = series.dropna().values
    if len(arr) < 20:
        return {
            "adf": None,
            "kpss": None,
            "is_stationary": None,
            "interpretation": f"Insufficient data (n={len(arr)}, need ≥20)",
        }

    result = {"adf": {}, "kpss": {}}

    try:
        # ADF test (null: unit root / non-stationary)
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(arr, autolag="AIC")
        result["adf"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "lags_used": int(adf_lags),
            "reject_null": adf_p < alpha,  # Reject = stationary
        }
    except Exception as e:
        result["adf"] = {"error": str(e)}

    try:
        # KPSS test (null: stationary)
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(arr, regression="c", nlags="auto")
        result["kpss"] = {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "lags_used": int(kpss_lags),
            "reject_null": kpss_p < alpha,  # Reject = non-stationary
        }
    except Exception as e:
        result["kpss"] = {"error": str(e)}

    # Combined interpretation
    adf_stationary = result["adf"].get("reject_null")
    kpss_stationary = not result["kpss"].get("reject_null", True)  # KPSS null is stationary

    if adf_stationary and kpss_stationary:
        conclusion = "stationary"
        interpretation = "Series is stationary (both tests agree)"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "non_stationary"
        interpretation = "Series is non-stationary (both tests agree) - consider differencing"
    elif adf_stationary and not kpss_stationary:
        conclusion = "difference_stationary"
        interpretation = "Series is difference-stationary (ADF rejects, KPSS rejects)"
    else:
        conclusion = "trend_stationary"
        interpretation = "Series is trend-stationary (ADF fails, KPSS passes)"

    result["is_stationary"] = conclusion == "stationary"
    result["conclusion"] = conclusion
    result["interpretation"] = interpretation

    return result


def decompose_time_series(
    series: pd.Series,
    period: Optional[int] = None,
    model: str = "additive"
) -> Dict[str, Any]:
    """Decompose time series into trend, seasonal, and residual components."""
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed"}

    arr = series.dropna()
    if len(arr) < 20:
        return {"error": f"Insufficient data (n={len(arr)}, need ≥20)"}

    # Auto-detect period if not provided
    if period is None:
        n = len(arr)
        if n >= 730:  # ~2 years daily
            period = 365
        elif n >= 104:  # ~2 years weekly
            period = 52
        elif n >= 24:  # ~2 years monthly
            period = 12
        else:
            period = max(2, n // 4)

    # Ensure period is valid
    period = min(period, len(arr) // 2)
    if period < 2:
        return {"error": "Period too small for decomposition"}

    try:
        decomposition = seasonal_decompose(arr, model=model, period=period)

        return {
            "period": period,
            "model": model,
            "trend_strength": float(1 - np.nanvar(decomposition.resid) / np.nanvar(arr - decomposition.seasonal)),
            "seasonal_strength": float(1 - np.nanvar(decomposition.resid) / np.nanvar(arr - decomposition.trend)),
            "trend_summary": {
                "mean": float(np.nanmean(decomposition.trend)),
                "start": float(decomposition.trend.dropna().iloc[0]) if len(decomposition.trend.dropna()) > 0 else None,
                "end": float(decomposition.trend.dropna().iloc[-1]) if len(decomposition.trend.dropna()) > 0 else None,
            },
            "residual_summary": {
                "mean": float(np.nanmean(decomposition.resid)),
                "std": float(np.nanstd(decomposition.resid)),
            },
            "interpretation": f"Decomposed with period={period}. "
                + f"Trend strength: {1 - np.nanvar(decomposition.resid) / np.nanvar(arr - decomposition.seasonal):.2f}, "
                + f"Seasonal strength: {1 - np.nanvar(decomposition.resid) / np.nanvar(arr - decomposition.trend):.2f}",
        }
    except Exception as e:
        return {"error": f"Decomposition failed: {e}"}


def generate_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = None,
    rolling_windows: List[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate lag and rolling window features for time series.

    Returns tuple of (df_with_features, new_column_names).
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]  # Default lags
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]  # Default windows

    df = df.copy()
    new_cols = []

    if target_col not in df.columns:
        return df, []

    series = df[target_col]

    # Lag features
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = series.shift(lag)
        new_cols.append(col_name)

    # Rolling statistics
    for window in rolling_windows:
        # Rolling mean
        col_mean = f"{target_col}_rolling_mean_{window}"
        df[col_mean] = series.rolling(window=window, min_periods=1).mean()
        new_cols.append(col_mean)

        # Rolling std
        col_std = f"{target_col}_rolling_std_{window}"
        df[col_std] = series.rolling(window=window, min_periods=2).std()
        new_cols.append(col_std)

        # Rolling min/max
        col_min = f"{target_col}_rolling_min_{window}"
        df[col_min] = series.rolling(window=window, min_periods=1).min()
        new_cols.append(col_min)

        col_max = f"{target_col}_rolling_max_{window}"
        df[col_max] = series.rolling(window=window, min_periods=1).max()
        new_cols.append(col_max)

    return df, new_cols


def forecast_arima(
    series: pd.Series,
    horizon: int = 10,
    order: Tuple[int, int, int] = None,
) -> Dict[str, Any]:
    """Forecast using ARIMA model.

    Auto-selects order if not provided (simple heuristic).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not installed"}

    arr = series.dropna().values
    if len(arr) < 30:
        return {"error": f"Insufficient data (n={len(arr)}, need ≥30)"}

    # Simple order selection if not provided
    if order is None:
        # Check stationarity
        stat_result = test_stationarity(pd.Series(arr))
        d = 0 if stat_result.get("is_stationary") else 1
        order = (2, d, 2)  # Simple default

    try:
        model = ARIMA(arr, order=order)
        fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(steps=horizon)
        conf_int = fitted.get_forecast(steps=horizon).conf_int()

        return {
            "order": order,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "forecast": [float(f) for f in forecast],
            "confidence_lower": [float(c) for c in conf_int[:, 0]],
            "confidence_upper": [float(c) for c in conf_int[:, 1]],
            "horizon": horizon,
            "interpretation": f"ARIMA{order} forecast for {horizon} periods. AIC={fitted.aic:.2f}",
        }
    except Exception as e:
        return {"error": f"ARIMA failed: {e}"}


def run_time_series_analysis(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run comprehensive time series analysis.

    Returns:
    - time_detection: Detected time columns
    - frequency: Detected frequency
    - stationarity: Stationarity test results
    - decomposition: Trend/seasonal decomposition
    - features_generated: Lag/rolling features created
    """
    config = config or {}
    results: Dict[str, Any] = {
        "time_detection": {},
        "frequency": {},
        "stationarity": {},
        "decomposition": {},
        "features": {},
        "summary": {},
    }

    # 1. Detect time columns if not provided
    if time_col is None:
        detection = detect_time_columns(df)
        results["time_detection"] = detection
        time_col = detection.get("best_candidate")

    if time_col is None:
        results["summary"]["error"] = "No time column detected"
        return results

    # 2. Parse time column and detect frequency
    try:
        time_series = pd.to_datetime(df[time_col], errors="coerce")
        results["frequency"] = detect_frequency(time_series)
    except Exception as e:
        results["frequency"] = {"error": str(e)}

    # 3. Analyze target column if provided
    if target_col and target_col in df.columns:
        target_series = df[target_col]

        if np.issubdtype(target_series.dtype, np.number):
            # Sort by time for proper analysis
            if time_col in df.columns:
                try:
                    sorted_df = df.sort_values(time_col)
                    target_series = sorted_df[target_col]
                except Exception:
                    pass

            # Stationarity test
            results["stationarity"] = test_stationarity(target_series)

            # Decomposition
            results["decomposition"] = decompose_time_series(target_series)

            # Generate lag features
            _, new_features = generate_lag_features(df, target_col)
            results["features"] = {
                "n_features_generated": len(new_features),
                "feature_names": new_features[:10],  # First 10
            }

    # Summary
    is_stationary = results.get("stationarity", {}).get("is_stationary")
    freq = results.get("frequency", {}).get("interpretation")

    results["summary"] = {
        "time_column": time_col,
        "target_column": target_col,
        "frequency": freq,
        "is_stationary": is_stationary,
        "recommendation": (
            "Data is stationary - can use standard ML models"
            if is_stationary
            else "Data is non-stationary - consider differencing or using ARIMA/Prophet"
        ) if is_stationary is not None else "Unable to determine stationarity",
    }

    return results

