import json
from typing import Dict, Any

from ..core.logs import model_decision
from ..core.llm import get_llm_client

# JSON Schema for structured LLM output
ROUTER_DECISIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of analysis steps"
        },
        "analysis_type": {
            "type": "string",
            "enum": ["predictive", "causal", "time_series", "statistical", "exploratory"],
            "description": "Primary analysis type based on user question"
        },
        "decisions": {
            "type": "object",
            "properties": {
                "class_weight": {
                    "type": "string",
                    "enum": ["balanced", "none"],
                    "description": "Use 'balanced' for imbalanced classification (>3:1 ratio)"
                },
                "metric": {
                    "type": "string",
                    "enum": ["f1", "accuracy", "roc_auc", "r2", "rmse"],
                    "description": "Primary evaluation metric"
                },
                "split": {
                    "type": "string",
                    "enum": ["random", "time", "stratified"],
                    "description": "Train/test split strategy"
                },
                "budget": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "Compute budget for model training"
                },
                "scaling": {
                    "type": "string",
                    "enum": ["standard", "robust", "minmax", "none"],
                    "description": "Feature scaling method"
                },
                "treatment": {
                    "type": "string",
                    "description": "Treatment variable for causal analysis (if applicable)"
                },
                "outcome": {
                    "type": "string",
                    "description": "Outcome variable for causal analysis (if applicable)"
                },
                "confounders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Confounder variables for causal analysis (columns that affect both treatment and outcome)"
                }
            },
            "required": ["class_weight", "metric", "split"]
        }
    },
    "required": ["plan", "analysis_type", "decisions"]
}

ROUTER_SYSTEM_PROMPT = """You are an expert data science planner. Analyze the user's question and dataset to return a JSON execution plan.

Return ONLY valid JSON with this exact structure:
{
  "plan": ["step1", "step2", ...],
  "analysis_type": "predictive" | "causal" | "time_series" | "statistical" | "exploratory",
  "decisions": {
    "class_weight": "balanced" | "none",
    "metric": "f1" | "accuracy" | "roc_auc" | "r2" | "rmse",
    "split": "random" | "time" | "stratified",
    "budget": "low" | "normal" | "high",
    "scaling": "standard" | "robust" | "minmax" | "none",
    "treatment": "column_name" (only for causal analysis),
    "outcome": "column_name" (only for causal analysis),
    "confounders": ["col1", "col2"] (only for causal analysis - variables that affect both treatment and outcome)
  }
}

## ANALYSIS TYPE DETECTION (choose one):

### predictive (default):
- Questions asking to "predict", "classify", "model", "forecast outcome"
- Standard ML classification or regression tasks

### causal:
- Questions about "effect of X on Y", "does X cause Y", "impact of treatment"
- Requires treatment and outcome variables to be specified
- If user mentions "control for" or "confounders", extract those column names
- Examples: "What is the effect of marketing spend on sales?", "Does training improve retention?"

### time_series:
- Questions about "forecast", "trend", "seasonal patterns", "future values"
- Data has a time/date column
- Examples: "Forecast next month's sales", "What's the trend?"

### statistical:
- Questions about "significant difference", "hypothesis test", "correlation"
- Testing relationships without building predictive models
- Examples: "Is there a significant difference between groups?", "Are X and Y correlated?"

### exploratory:
- General questions about data patterns, distributions, summaries
- No specific prediction or causal question
- Examples: "What patterns exist?", "Summarize this data"

## DECISION RULES:

### class_weight:
- "balanced": Minority class <25% (imbalance >3:1), fraud/churn detection
- "none": Classes roughly equal (40-60% split)

### metric:
- "f1": Binary classification, especially imbalanced
- "roc_auc": Ranking/probability important
- "accuracy": Balanced multi-class
- "r2"/"rmse": Regression

### split:
- "stratified": Classification (preserves class distribution)
- "time": Time series or when temporal ordering matters
- "random": Regression or no temporal order

### budget:
- "low": <1000 rows, "normal": 1000-100k rows, "high": >100k rows

### scaling:
- "robust": Outliers/high skewness, "standard": Normal distributions, "none": Tree models only

## EXAMPLES:

Example 1 - Fraud Detection (predictive, imbalanced):
{"plan": ["eda", "feature_engineering", "modeling"], "analysis_type": "predictive", "decisions": {"class_weight": "balanced", "metric": "f1", "split": "stratified", "budget": "normal", "scaling": "robust"}}

Example 2 - Causal Effect (causal):
Question: "What is the effect of discount on purchase? Control for age and income."
{"plan": ["eda", "causal_analysis", "sensitivity"], "analysis_type": "causal", "decisions": {"class_weight": "none", "metric": "f1", "split": "random", "budget": "normal", "scaling": "standard", "treatment": "discount", "outcome": "purchase", "confounders": ["age", "income"]}}

Example 3 - Time Series Forecast:
{"plan": ["eda", "stationarity_test", "forecasting"], "analysis_type": "time_series", "decisions": {"class_weight": "none", "metric": "rmse", "split": "time", "budget": "normal", "scaling": "standard"}}

Example 4 - Statistical Test:
Question: "Is there a significant difference in salary between departments?"
{"plan": ["eda", "statistical_tests"], "analysis_type": "statistical", "decisions": {"class_weight": "none", "metric": "f1", "split": "random", "budget": "low", "scaling": "none"}}"""


def build_context_pack(eda: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured context pack for the LLM router."""
    target = manifest.get("target") or eda.get("target")
    nunique = eda.get("nunique") or {}

    # Try to get class distribution from EDA
    class_distribution = None
    imbalance_hint = None
    target_dist = eda.get("target_distribution")
    if target_dist and isinstance(target_dist, dict):
        class_distribution = target_dist
        # Calculate imbalance ratio
        values = list(target_dist.values())
        if len(values) == 2:
            ratio = max(values) / max(min(values), 1)
            if ratio > 3:
                imbalance_hint = f"IMBALANCED (ratio={ratio:.1f}:1) - use class_weight=balanced"
            else:
                imbalance_hint = f"balanced (ratio={ratio:.1f}:1)"
    elif target and target in nunique:
        target_nunique = nunique.get(target, 0)
        if target_nunique == 2:
            imbalance_hint = "binary_classification - check for imbalance"

    return {
        "shape": eda.get("shape"),
        "dtypes": eda.get("dtypes"),
        "missing_summary": {
            k: v.get("pct") for k, v in (eda.get("missing") or {}).items()
        },
        "nunique": nunique,
        "target": target,
        "id_candidates": eda.get("id_candidates"),
        "constant_columns": eda.get("constant_columns"),
        "top_correlations": eda.get("top_correlations"),
        "rare_levels": {k: len(v) for k, v in (eda.get("rare_levels") or {}).items()},
        "time_columns": eda.get("time_columns") or eda.get("time_like_candidates"),
        "recommendations": eda.get("recommendations"),
        "question": manifest.get("question"),
        "business_context": manifest.get("nl_description") or manifest.get("context"),
        "imbalance_hint": imbalance_hint,
        "class_distribution": class_distribution,
        "skew": eda.get("skew"),  # For outlier detection
        "kurtosis": eda.get("kurtosis"),
    }


def _build_user_prompt(ctx: Dict[str, Any]) -> str:
    """Build a structured user prompt for the router LLM."""
    shape = ctx.get("shape") or {}
    rows = shape.get("rows", "unknown")
    cols = shape.get("cols", "unknown")

    # Build missing summary
    missing = ctx.get("missing_summary") or {}
    high_missing = [f"{k}:{v:.0%}" for k, v in missing.items() if v and v > 0.1]
    missing_str = ", ".join(high_missing[:5]) if high_missing else "none significant"

    # Target info
    target = ctx.get("target") or "unknown"
    nunique = ctx.get("nunique") or {}
    target_nunique = nunique.get(target, "unknown")

    # Task type inference
    task_type = "unknown"
    imbalance_info = "unknown"
    if isinstance(target_nunique, int):
        if target_nunique == 2:
            task_type = "binary_classification"
            # Check for imbalance hint
            imbalance_hint = ctx.get("imbalance_hint")
            class_dist = ctx.get("class_distribution")
            if class_dist:
                imbalance_info = f"class distribution: {class_dist}"
            elif imbalance_hint:
                imbalance_info = str(imbalance_hint)
            else:
                imbalance_info = "check for imbalance (use balanced if minority <25%)"
        elif target_nunique <= 10:
            task_type = "multiclass_classification"
            imbalance_info = f"{target_nunique} classes"
        else:
            task_type = "regression"
            imbalance_info = "not applicable"

    # Time columns
    time_cols = ctx.get("time_columns") or []
    time_str = ", ".join(time_cols[:3]) if time_cols else "none"

    # ID candidates
    id_cols = ctx.get("id_candidates") or []
    id_str = ", ".join(id_cols[:3]) if id_cols else "none"

    # Top correlations with target
    correlations = ctx.get("top_correlations") or []
    corr_with_target = [c for c in correlations if target in (c[0], c[1])][:3]
    corr_str = ", ".join([f"{c[0]}-{c[1]}:{c[2]:.2f}" for c in corr_with_target]) if corr_with_target else "none"

    # Outliers hint
    outlier_hint = "unknown"
    skew = ctx.get("skew") or {}
    if skew:
        high_skew = [k for k, v in skew.items() if abs(v) > 2]
        if high_skew:
            outlier_hint = f"high skewness in {len(high_skew)} features (use robust scaling)"
        else:
            outlier_hint = "no significant skewness"

    return f"""Analyze this dataset and return the execution plan JSON.

## Dataset Summary:
- Shape: {rows} rows × {cols} columns
- Task type: {task_type}
- Target column: "{target}" ({target_nunique} unique values)
- Class imbalance: {imbalance_info}
- High missing (>10%): {missing_str}
- Time columns: {time_str}
- ID candidates (exclude from modeling): {id_str}
- Target correlations: {corr_str}
- Outliers/skewness: {outlier_hint}

## User Request:
Question: {ctx.get("question") or "Predict the target"}
Business context: {ctx.get("business_context") or "Not provided"}

## EDA Recommendations:
{ctx.get("recommendations") or ["No specific recommendations"]}

Based on the above, return the JSON plan with appropriate decisions.
Remember: Use class_weight="balanced" if this looks like fraud/churn/rare-event detection or if minority class <25%."""


def plan_with_router(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Plan the analysis using LLM with structured output."""
    fallback = {
        "plan": ["feature_expert", "modeling", "evaluation"],
        "analysis_type": "predictive",
        "decisions": {"budget": "normal", "class_weight": "none", "metric": "f1", "split": "stratified"},
        "source": "fallback",
    }

    client = get_llm_client()
    if client is None:
        if ctx.get("job_id"):
            model_decision(
                str(ctx.get("job_id")),
                "Router: No LLM provider available; using fallback",
            )
        return fallback

    try:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(ctx)},
        ]

        # Use structured JSON output mode
        txt = client.chat(
            messages=messages,
            temperature=0.1,
            json_mode=True,
            json_schema=ROUTER_DECISIONS_SCHEMA,
        )

        # With json_mode=True, response should be clean JSON (no markdown)
        # But still handle edge cases
        txt = txt.strip().removeprefix("```json").removesuffix("```").strip()

        sanitized = txt.replace("\n", " ")[:1000]
        job_id = str(ctx.get("job_id") or "")
        if job_id:
            model_decision(job_id, f"Router raw plan ({client.provider_name}): {sanitized}")

        plan = json.loads(txt)

        # Validate and normalize analysis_type
        valid_types = ["predictive", "causal", "time_series", "statistical", "exploratory"]
        analysis_type = plan.get("analysis_type", "predictive")
        if analysis_type not in valid_types:
            analysis_type = "predictive"
        plan["analysis_type"] = analysis_type

        # Validate and normalize decisions
        decisions = plan.get("decisions") or {}
        normalized = {
            "class_weight": decisions.get("class_weight", "none"),
            "metric": decisions.get("metric", "f1"),
            "split": decisions.get("split", "stratified"),
            "budget": decisions.get("budget", "normal"),
            "scaling": decisions.get("scaling", "standard"),
        }

        # Include causal-specific fields if present
        if analysis_type == "causal":
            if decisions.get("treatment"):
                normalized["treatment"] = decisions["treatment"]
            if decisions.get("outcome"):
                normalized["outcome"] = decisions["outcome"]

        plan["decisions"] = normalized
        plan["source"] = client.provider_name

        if job_id:
            model_decision(job_id, f"Router analysis_type: {analysis_type}, decisions: {json.dumps(normalized)}")

        return plan
    except Exception as e:
        if ctx.get("job_id"):
            model_decision(
                str(ctx.get("job_id")), f"Router failed: {e}; using fallback"
            )
        return fallback
