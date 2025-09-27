# AI Data Scientist – Quick Cookbook

This document summarizes common controls and how to steer the agent.

## Profiles
- profile: "lean" – minimal extras (skips time-series/text FE and fairness). Faster, simpler.
- profile: "full" – enables all enrichments (default).

How to set: add to manifest (router plan or client payload)
```
{
  "profile": "lean"
}
```

## Router decisions (examples)
- metric: "f1" | "accuracy" | "roc_auc" | "pr_auc" | "r2" | "rmse"
- split: "time" | "random"
- budget: "low" | "normal" | "high" (impacts search/HPO effort)
- class_weight: "balanced" | null
- calibration: true | false
- timeseries_fe: true | false
- lag_windows: [3, 7] (time-series)

## Feature Engineering
- Datetime features: always safe adds (year, month, day, dow, hour when applicable)
- Time-series features: enabled by profile!=lean and router; adds lag1 and rolling means (shift(1) for leakage safety)
- Text features: enabled by profile!=lean; lightweight length/character counts for up to 5 columns

## Fairness/Slice Metrics
- Computes slice prevalence (classification) or per-group target mean (regression)
- If model test predictions are available, also computes per-slice accuracy/f1 on test fold
- Artifacts: jobs/{job_id}/fairness.json

## Reproducibility
- Records environment versions and dataset fingerprint; artifact at jobs/{job_id}/reproducibility.json

## Experiments Registry
- Append-only CSV at data/experiments.csv capturing job_id, dataset hash, task, best model, metric, desired metric, profile

## HPO / Search Effort
- The router budget influences randomized search iterations (up to 10) for the quick sweep
  - low → 0 (disabled), normal → ~4, high → up to 10

## Returning Predictions
- The pipeline attaches test-fold predictions under result.modeling.pred_test (best-effort)
  - keys: index, y_true, y_pred, proba (optional)
  - Used to compute per-slice performance in fairness

## Tips
- For time series, set split: "time" and consider lag_windows; the agent will use TimeSeriesSplit.
- For imbalanced classification, set class_weight: "balanced" and consider calibration: true.
- Use profile: "lean" for quick iteration; switch to "full" for deeper artifacts (fairness, text/TS FE).

