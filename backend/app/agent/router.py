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
                }
            },
            "required": ["class_weight", "metric", "split"]
        }
    },
    "required": ["plan", "decisions"]
}

ROUTER_SYSTEM_PROMPT = """You are an expert data science planner. Analyze dataset characteristics and return a JSON execution plan.

Return ONLY valid JSON with this exact structure:
{
  "plan": ["step1", "step2", ...],
  "decisions": {
    "class_weight": "balanced" | "none",
    "metric": "f1" | "accuracy" | "roc_auc" | "r2" | "rmse",
    "split": "random" | "time" | "stratified",
    "budget": "low" | "normal" | "high",
    "scaling": "standard" | "robust" | "minmax" | "none"
  }
}

## DECISION RULES (follow strictly):

### class_weight:
- "balanced": Use when minority class <25% of data (imbalance ratio >3:1)
- "balanced": Use for fraud detection, churn prediction, rare event detection
- "none": Use when classes are roughly equal (40-60% split)

### metric:
- "f1": Default for binary classification, especially imbalanced data
- "roc_auc": When ranking/probability is important (e.g., credit scoring)
- "accuracy": Only for balanced multi-class classification
- "r2": For regression tasks (numeric target with many unique values)
- "rmse": For regression when error magnitude matters (e.g., pricing)

### split:
- "stratified": Default for classification (preserves class distribution)
- "time": When time_columns exist AND data should not leak future info
- "random": For regression or when no temporal ordering matters

### budget:
- "low": <1000 rows - use simpler models, less hyperparameter tuning
- "normal": 1000-100k rows - standard model suite
- "high": >100k rows - can afford expensive models, more tuning

### scaling:
- "standard": Default - StandardScaler for normally distributed features
- "robust": When outliers present (high skewness/kurtosis in data)
- "minmax": When features need to be in [0,1] range
- "none": When only using tree-based models

## EXAMPLES:

Example 1 - Fraud Detection (imbalanced):
Input: 10000 rows, binary target with 2% fraud rate
Output: {"plan": ["eda", "feature_engineering", "modeling", "evaluation"], "decisions": {"class_weight": "balanced", "metric": "f1", "split": "stratified", "budget": "normal", "scaling": "robust"}}

Example 2 - House Price Prediction (regression):
Input: 5000 rows, numeric target (prices), some outliers
Output: {"plan": ["eda", "feature_engineering", "modeling", "evaluation"], "decisions": {"class_weight": "none", "metric": "r2", "split": "random", "budget": "normal", "scaling": "robust"}}

Example 3 - Customer Churn (balanced classification):
Input: 2000 rows, binary target ~45%/55% split
Output: {"plan": ["eda", "modeling", "evaluation"], "decisions": {"class_weight": "none", "metric": "f1", "split": "stratified", "budget": "normal", "scaling": "standard"}}

Example 4 - Time Series Classification:
Input: 50000 rows, has date column, binary target
Output: {"plan": ["eda", "feature_engineering", "modeling", "evaluation"], "decisions": {"class_weight": "balanced", "metric": "roc_auc", "split": "time", "budget": "high", "scaling": "standard"}}"""


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

        # Validate and normalize decisions
        decisions = plan.get("decisions") or {}
        normalized = {
            "class_weight": decisions.get("class_weight", "none"),
            "metric": decisions.get("metric", "f1"),
            "split": decisions.get("split", "stratified"),
            "budget": decisions.get("budget", "normal"),
            "scaling": decisions.get("scaling", "standard"),
        }
        plan["decisions"] = normalized
        plan["source"] = client.provider_name

        if job_id:
            model_decision(job_id, f"Router decisions: {json.dumps(normalized)}")

        return plan
    except Exception as e:
        if ctx.get("job_id"):
            model_decision(
                str(ctx.get("job_id")), f"Router failed: {e}; using fallback"
            )
        return fallback
