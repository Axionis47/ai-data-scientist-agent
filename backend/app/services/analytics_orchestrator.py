"""Analytics Orchestrator Service

Routes analysis requests to the appropriate specialized service based on
question intent and data characteristics.

Public API:
- detect_analysis_type(question, df) -> Dict with analysis type and confidence
- run_analysis(df, analysis_type, config) -> Dict with results
- run_comprehensive_analysis(df, question, manifest) -> Dict with all applicable analyses

Analysis Types:
- "predictive": Standard ML modeling (classification/regression)
- "causal": Causal inference and treatment effects
- "time_series": Time series analysis and forecasting
- "statistical": Hypothesis testing and statistical inference
- "exploratory": EDA and descriptive statistics
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import re
import numpy as np
import pandas as pd

from .statistical_testing_service import run_statistical_tests
from .time_series_service import run_time_series_analysis, detect_time_columns
from .causal_inference_service import run_causal_analysis, detect_causal_question


# Analysis type detection patterns
ANALYSIS_PATTERNS = {
    "causal": {
        "keywords": ["effect", "cause", "impact", "treatment", "intervention", "causal"],
        "patterns": [
            r"effect\s+of\s+\w+\s+on",
            r"does\s+\w+\s+cause",
            r"how\s+does\s+\w+\s+affect",
            r"what\s+happens\s+if",
        ],
        "weight": 1.0,
    },
    "time_series": {
        "keywords": ["forecast", "predict future", "trend", "seasonal", "time series", "arima", "prophet"],
        "patterns": [
            r"forecast\s+\w+",
            r"predict\s+(?:future|next|coming)",
            r"time\s+series",
            r"trend\s+analysis",
        ],
        "weight": 0.9,
    },
    "statistical": {
        "keywords": ["significant", "hypothesis", "test", "p-value", "correlation", "difference between"],
        "patterns": [
            r"is\s+there\s+a\s+(?:significant\s+)?difference",
            r"test\s+(?:if|whether)",
            r"hypothesis",
            r"are\s+\w+\s+and\s+\w+\s+correlated",
        ],
        "weight": 0.8,
    },
    "predictive": {
        "keywords": ["predict", "classify", "model", "machine learning", "regression", "classification"],
        "patterns": [
            r"predict\s+\w+",
            r"classify",
            r"build\s+a?\s*model",
        ],
        "weight": 0.7,
    },
}


def detect_analysis_type(
    question: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Detect the most appropriate analysis type based on question and data."""
    question_lower = question.lower()
    scores = {}
    
    # Score each analysis type
    for analysis_type, config in ANALYSIS_PATTERNS.items():
        score = 0.0
        matches = []
        
        # Keyword matching
        for kw in config["keywords"]:
            if kw in question_lower:
                score += 0.3 * config["weight"]
                matches.append(f"keyword: {kw}")
        
        # Pattern matching
        for pattern in config["patterns"]:
            if re.search(pattern, question_lower):
                score += 0.5 * config["weight"]
                matches.append(f"pattern: {pattern}")
        
        scores[analysis_type] = {"score": score, "matches": matches}
    
    # Data-based signals (if dataframe provided)
    if df is not None:
        # Time series signals
        time_detection = detect_time_columns(df)
        if time_detection.get("n_candidates", 0) > 0:
            scores["time_series"]["score"] += 0.2
            scores["time_series"]["matches"].append("time columns detected")
        
        # Statistical signals
        n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
        n_num = len(df.select_dtypes(include=[np.number]).columns)
        if n_cat >= 2 and n_num >= 1:
            scores["statistical"]["score"] += 0.1
    
    # Find best match
    best_type = max(scores, key=lambda x: scores[x]["score"])
    best_score = scores[best_type]["score"]
    
    # Default to predictive if no strong signal
    if best_score < 0.2:
        best_type = "predictive"
        best_score = 0.5
    
    return {
        "analysis_type": best_type,
        "confidence": min(best_score, 1.0),
        "all_scores": {k: v["score"] for k, v in scores.items()},
        "matches": scores[best_type]["matches"],
        "recommendation": _get_recommendation(best_type),
    }


def _get_recommendation(analysis_type: str) -> str:
    """Get a recommendation message for the detected analysis type."""
    recommendations = {
        "causal": "Will perform causal inference to estimate treatment effects",
        "time_series": "Will analyze time series patterns and generate forecasts",
        "statistical": "Will run hypothesis tests and statistical analyses",
        "predictive": "Will build predictive models (classification/regression)",
        "exploratory": "Will perform exploratory data analysis",
    }
    return recommendations.get(analysis_type, "Will perform standard analysis")


def run_analysis(
    df: pd.DataFrame,
    analysis_type: str,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run the specified type of analysis."""
    config = config or {}
    target = config.get("target")
    treatment = config.get("treatment")
    time_col = config.get("time_col")

    if analysis_type == "causal":
        outcome = config.get("outcome") or target
        return run_causal_analysis(
            df,
            treatment=treatment,
            outcome=outcome,
            confounders=config.get("confounders"),
            config=config,
        )

    elif analysis_type == "time_series":
        return run_time_series_analysis(
            df,
            time_col=time_col,
            target_col=target,
            config=config,
        )

    elif analysis_type == "statistical":
        return run_statistical_tests(
            df,
            target=target,
            config=config,
        )

    else:
        # Default: return basic info
        return {
            "analysis_type": analysis_type,
            "message": f"Analysis type '{analysis_type}' will use standard pipeline",
        }


def run_comprehensive_analysis(
    df: pd.DataFrame,
    question: str,
    manifest: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run comprehensive analysis including all applicable analysis types.

    This is the main entry point that:
    1. Detects the primary analysis type
    2. Runs the primary analysis
    3. Optionally runs supplementary analyses
    """
    manifest = manifest or {}
    results: Dict[str, Any] = {
        "question": question,
        "analysis_detection": {},
        "primary_analysis": {},
        "supplementary_analyses": {},
        "summary": {},
    }

    # 1. Detect analysis type
    detection = detect_analysis_type(question, df)
    results["analysis_detection"] = detection
    primary_type = detection["analysis_type"]

    # Extract config from manifest
    config = {
        "target": manifest.get("target") or manifest.get("framing", {}).get("target"),
        "treatment": manifest.get("treatment"),
        "outcome": manifest.get("outcome"),
        "time_col": manifest.get("time_col"),
    }

    # 2. Run primary analysis
    results["primary_analysis"] = run_analysis(df, primary_type, config)
    results["primary_analysis"]["analysis_type"] = primary_type

    # 3. Run supplementary statistical tests (always useful)
    if primary_type != "statistical":
        try:
            stat_results = run_statistical_tests(df, target=config.get("target"))
            # Only include summary to avoid bloat
            results["supplementary_analyses"]["statistical_summary"] = stat_results.get("summary", {})
        except Exception:
            pass

    # 4. Detect time series if not primary
    if primary_type != "time_series":
        try:
            time_detection = detect_time_columns(df)
            if time_detection.get("n_candidates", 0) > 0:
                results["supplementary_analyses"]["time_detection"] = {
                    "time_columns_found": time_detection.get("n_candidates"),
                    "best_candidate": time_detection.get("best_candidate"),
                    "suggestion": "Consider time series analysis if temporal patterns are important",
                }
        except Exception:
            pass

    # Summary
    results["summary"] = {
        "primary_analysis_type": primary_type,
        "confidence": detection.get("confidence"),
        "n_supplementary_analyses": len(results["supplementary_analyses"]),
        "recommendation": detection.get("recommendation"),
    }

    return results


# Feature flags for analysis types
CAUSAL_ANALYSIS_ENABLED = True
TIME_SERIES_ANALYSIS_ENABLED = True
STATISTICAL_TESTING_ENABLED = True


def get_available_analyses() -> Dict[str, bool]:
    """Return which analysis types are available."""
    from .causal_inference_service import DOWHY_AVAILABLE
    from .time_series_service import STATSMODELS_AVAILABLE, PROPHET_AVAILABLE

    return {
        "causal_inference": CAUSAL_ANALYSIS_ENABLED,
        "causal_inference_advanced": DOWHY_AVAILABLE,
        "time_series": TIME_SERIES_ANALYSIS_ENABLED,
        "time_series_advanced": STATSMODELS_AVAILABLE,
        "time_series_prophet": PROPHET_AVAILABLE,
        "statistical_testing": STATISTICAL_TESTING_ENABLED,
    }

