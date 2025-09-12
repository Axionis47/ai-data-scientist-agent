# Modeling module (app/modeling/pipeline.py)

Entry point
- run_modeling(job_id, df, eda, manifest) -> modeling dict

What it does
- Determines task (classification/regression) from target
- Preprocesses: impute numeric, TopK rare handling for categoricals, OHE
- Generates candidate models (linear, random forest, HGB)
- Applies safe CV folds for tiny datasets
- Optional quick hyperparameter search (small, time-capped)
- Produces leaderboard/best and explain artifacts (importances, ROC/PR, diagnostics)
- Writes modeling.json for resumability

Key helpers
- TopKCategorical: keeps top-K frequent levels, maps rest to __OTHER__
- quick_search: small randomized search over narrow grids when enabled

Outputs
- modeling dict with keys: task, features, selected_tools, leaderboard[], best, explain

