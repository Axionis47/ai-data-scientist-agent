"""
Causal Readiness Gate.

Runs diagnostic checks and determines if causal estimation can proceed.
Returns structured CausalReadinessReport with PASS/WARN/FAIL status.

Thresholds and rules are documented inline for transparency.
"""

from pathlib import Path

from packages.contracts.models import (
    CausalReadinessReport,
    CausalSpecArtifact,
    DiagnosticArtifact,
)

from .tools_causal import (
    balance_check_smd,
    check_leakage,
    check_missingness,
    check_positivity_overlap,
    check_time_ordering,
    check_treatment_type,
    infer_candidate_columns,
)

# =============================================================================
# Standard follow-up questions for causal assumptions
# =============================================================================

ASSUMPTION_QUESTIONS = [
    "What is the assignment mechanism? (e.g., randomized, policy rule, self-selection)",
    "Is there potential for interference between units (violation of SUTVA)?",
    "Are there any unmeasured confounders you are aware of?",
    "What is your preferred approach for handling missing data?",
]


def _build_diagnostic_artifact(diag_dict: dict) -> DiagnosticArtifact:
    """Convert diagnostic dict to DiagnosticArtifact."""
    return DiagnosticArtifact(
        name=diag_dict.get("name", "unknown"),
        status=diag_dict.get("status", "FAIL"),
        details=diag_dict.get("details", {}),
        recommendations=diag_dict.get("recommendations", []),
    )


def causal_readiness_gate(  # noqa: PLR0915
    question: str,
    doc_id: str,  # noqa: ARG001 - reserved for future RAG context
    dataset_id: str | None,
    column_names: list[str],
    inferred_types: dict[str, str],
    datasets_dir: Path,
    spec_override: dict | None = None,
) -> CausalReadinessReport:
    """
    Run causal readiness checks and return structured report.

    Args:
        question: User's causal question
        doc_id: Document ID for context
        dataset_id: Dataset ID (required for causal analysis)
        column_names: List of column names in dataset
        inferred_types: Column type mapping
        datasets_dir: Path to datasets directory
        spec_override: Optional user-provided specification override

    Returns:
        CausalReadinessReport with readiness_status, spec, diagnostics, and followup_questions
    """
    diagnostics: list[DiagnosticArtifact] = []
    followup_questions: list[str] = []

    # Initialize spec from override or infer
    treatment = spec_override.get("treatment") if spec_override else None
    outcome = spec_override.get("outcome") if spec_override else None
    unit = spec_override.get("unit") if spec_override else None
    time_col = spec_override.get("time_col") if spec_override else None
    horizon = spec_override.get("horizon") if spec_override else None
    confounders = spec_override.get("confounders", []) if spec_override else []

    # ==========================================================================
    # GATE 0: Dataset required
    # ==========================================================================
    if not dataset_id:
        spec = CausalSpecArtifact(
            treatment=treatment,
            outcome=outcome,
            questions=["Please upload a dataset to perform causal analysis."],
        )
        return CausalReadinessReport(
            readiness_status="FAIL",
            spec=spec,
            diagnostics=[],
            followup_questions=["Please upload a dataset to perform causal analysis."],
            ready_for_estimation=False,
        )

    # ==========================================================================
    # GATE 1: Infer treatment/outcome if not specified
    # ==========================================================================
    if not treatment or not outcome:
        inference = infer_candidate_columns(question, column_names, inferred_types)
        treatment_candidates = inference["treatment_candidates"]
        outcome_candidates = inference["outcome_candidates"]

        if not treatment and treatment_candidates:
            treatment = treatment_candidates[0]  # Take first candidate
        if not outcome and outcome_candidates:
            outcome = outcome_candidates[0]

        # If still missing, ask user
        if not treatment:
            followup_questions.append(
                f"Which column represents the treatment/intervention? Available columns: {column_names}"
            )
        if not outcome:
            followup_questions.append(
                f"Which column represents the outcome? Available columns: {column_names}"
            )

    # If we can't determine treatment/outcome, FAIL early
    if not treatment or not outcome:
        spec = CausalSpecArtifact(
            treatment=treatment,
            outcome=outcome,
            questions=followup_questions,
        )
        return CausalReadinessReport(
            readiness_status="FAIL",
            spec=spec,
            diagnostics=[],
            followup_questions=followup_questions,
            ready_for_estimation=False,
        )

    # ==========================================================================
    # GATE 2: Check treatment type (must be binary)
    # ==========================================================================
    treatment_check = check_treatment_type(dataset_id, treatment, datasets_dir)
    diag = treatment_check["diagnostic"]
    diagnostics.append(_build_diagnostic_artifact(diag))

    if diag["status"] == "FAIL":
        followup_questions.extend(diag.get("recommendations", []))
        spec = CausalSpecArtifact(
            treatment=treatment,
            outcome=outcome,
            unit=unit,
            time_col=time_col,
            questions=followup_questions,
        )
        return CausalReadinessReport(
            readiness_status="FAIL",
            spec=spec,
            diagnostics=diagnostics,
            followup_questions=followup_questions,
            ready_for_estimation=False,
        )

    if diag["status"] == "WARN":
        followup_questions.extend(diag.get("recommendations", []))

    # ==========================================================================
    # GATE 3: Check time ordering
    # ==========================================================================
    time_check = check_time_ordering(dataset_id, time_col, treatment, outcome, datasets_dir)
    diag = time_check["diagnostic"]
    diagnostics.append(_build_diagnostic_artifact(diag))

    if diag["status"] == "WARN":
        followup_questions.extend(diag.get("recommendations", []))

    # ==========================================================================
    # GATE 4: Infer confounders if not provided
    # ==========================================================================
    if not confounders:
        # Use all numeric columns except treatment/outcome as potential confounders
        confounders = [
            c for c in column_names
            if c not in [treatment, outcome, time_col, unit]
            and inferred_types.get(c) in ["int", "float", "bool", "category"]
        ]

    # ==========================================================================
    # GATE 5: Check for leakage in confounders
    # ==========================================================================
    if confounders:
        leakage_check = check_leakage(dataset_id, treatment, outcome, confounders, datasets_dir)
        diag = leakage_check["diagnostic"]
        diagnostics.append(_build_diagnostic_artifact(diag))

        if diag["status"] == "WARN":
            followup_questions.extend(diag.get("recommendations", []))
            # Remove suspicious columns from confounders
            suspicious = diag.get("details", {}).get("suspicious_columns", [])
            confounders = [c for c in confounders if c not in suspicious]

    # ==========================================================================
    # GATE 6: Check missingness
    # ==========================================================================
    cols_to_check = [treatment, outcome] + confounders
    missingness_check = check_missingness(dataset_id, cols_to_check, treatment, datasets_dir)
    diag = missingness_check["diagnostic"]
    diagnostics.append(_build_diagnostic_artifact(diag))

    if diag["status"] == "FAIL":
        followup_questions.extend(diag.get("recommendations", []))
        spec = CausalSpecArtifact(
            treatment=treatment,
            outcome=outcome,
            unit=unit,
            time_col=time_col,
            confounders_selected=confounders,
            questions=followup_questions,
        )
        return CausalReadinessReport(
            readiness_status="FAIL",
            spec=spec,
            diagnostics=diagnostics,
            followup_questions=followup_questions,
            ready_for_estimation=False,
        )

    if diag["status"] == "WARN":
        followup_questions.extend(diag.get("recommendations", []))

    # ==========================================================================
    # GATE 7: Check positivity/overlap (only if we have confounders)
    # ==========================================================================
    if confounders:
        positivity_check = check_positivity_overlap(dataset_id, treatment, confounders, datasets_dir)
        diag = positivity_check["diagnostic"]
        diagnostics.append(_build_diagnostic_artifact(diag))

        if diag["status"] == "FAIL":
            followup_questions.extend(diag.get("recommendations", []))
            spec = CausalSpecArtifact(
                treatment=treatment,
                outcome=outcome,
                unit=unit,
                time_col=time_col,
                confounders_selected=confounders,
                questions=followup_questions,
            )
            return CausalReadinessReport(
                readiness_status="FAIL",
                spec=spec,
                diagnostics=diagnostics,
                followup_questions=followup_questions,
                ready_for_estimation=False,
            )

        if diag["status"] == "WARN":
            followup_questions.extend(diag.get("recommendations", []))

    # ==========================================================================
    # GATE 8: Balance check (SMD)
    # ==========================================================================
    if confounders:
        balance_check = balance_check_smd(dataset_id, treatment, confounders, datasets_dir)
        diag = balance_check["diagnostic"]
        diagnostics.append(_build_diagnostic_artifact(diag))

        if diag["status"] in ["WARN", "FAIL"]:
            followup_questions.extend(diag.get("recommendations", []))

    # ==========================================================================
    # Always add assumption questions
    # ==========================================================================
    for q in ASSUMPTION_QUESTIONS:
        if q not in followup_questions:
            followup_questions.append(q)

    # ==========================================================================
    # Determine final readiness
    # ==========================================================================
    statuses = [d.status for d in diagnostics]

    if "FAIL" in statuses:
        readiness_status = "FAIL"
        ready_for_estimation = False
    elif "WARN" in statuses:
        readiness_status = "WARN"
        ready_for_estimation = False
    else:
        readiness_status = "PASS"
        ready_for_estimation = True

    # Build recommended estimators (Phase 4 will implement these)
    recommended_estimators = [
        "Inverse Probability Weighting (IPW)",
        "Augmented IPW (AIPW/Doubly Robust)",
        "Matching (Propensity Score)",
        "Regression Adjustment",
    ]

    # Build final spec
    spec = CausalSpecArtifact(
        treatment=treatment,
        outcome=outcome,
        unit=unit,
        time_col=time_col,
        horizon=horizon,
        confounders_selected=confounders,
        confounders_missing=[],  # Would be populated from domain knowledge
        assumptions=[
            "Unconfoundedness (conditional on selected confounders)",
            "Positivity (overlap in propensity scores)",
            "SUTVA (no interference between units)",
            "Consistency (well-defined treatment)",
        ],
        questions=followup_questions,
    )

    return CausalReadinessReport(
        readiness_status=readiness_status,
        spec=spec,
        diagnostics=diagnostics,
        followup_questions=followup_questions,
        ready_for_estimation=ready_for_estimation,
        recommended_estimators=recommended_estimators if ready_for_estimation else [],
    )

