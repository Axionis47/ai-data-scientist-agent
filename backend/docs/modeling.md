# Modelling module (app/modeling/pipeline.py)

Entry point
- run_modeling(job_id, df, eda, manifest) -> modelling dict

What it does
- Determines task (classification/regression) from the target
- Preprocesses: impute numeric, TopK rare handling for categoricals, OHE
- Generates candidate models (linear, random forest, HGB)
- Applies safe CV folds for tiny datasets
- Optional quick hyperparameter search (small, time-capped)
- Produces leaderboard/best and explain artefacts (importances, ROC/PR, diagnostics)
- Writes modeling.json for resumability

Key helpers
- TopKCategorical: keeps top-K frequent levels, maps the rest to __OTHER__
- quick_search: small randomised search over narrow grids when enabled

Outputs
- modelling dict with keys: task, features, selected_tools, leaderboard[], best, explain

Notes
- For very small datasets, expect simpler models and higher variance in metrics. That is normal.
