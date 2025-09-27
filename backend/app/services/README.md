# Services layer

This directory contains thin service facades that expose stable, testable entry points
around the core subsystems. The pipeline/orchestrator should import from these modules
rather than from the concrete implementation modules. This improves modularity,
facilitates process isolation, and makes it easier to migrate to separate services later.

Modules
- eda_service.py
  - infer_format, load_dataframe, load_sampled_chunked_csv, compute_eda,
    compute_target_relations, compute_timeseries_hints
- modeling_service.py
  - run_modeling
- router_service.py
  - build_context_pack, plan_with_router
- reporting_service.py
  - reporting_expert
- critique_service.py
  - CRITIQUE_POST_MODEL, critique_post_model

Contract
- Inputs/outputs mirror the underlying functions (documented in backend/docs/*)
- This layer does not add behavior; it stabilizes imports and enables future isolation.

