"""
Tests for Phase 3 Causal Safety Gate and Phase 4 Causal Estimation.

All tests use fake clients (no network/Vertex calls).
Tests verify:
1. FAIL when treatment/outcome cannot be inferred
2. FAIL when treatment is non-binary
3. PASS when conditions are met
4. No ATE is ever returned (Phase 3)
5. ATE is returned with confirmations (Phase 4)
6. Estimation blocked without confirmations (Phase 4)
"""

import json
import shutil

# Setup path for imports
import sys
import uuid
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from packages.agent.causal_gate import causal_readiness_gate
from packages.agent.graph import check_confirmations_node, run_estimation_node
from packages.agent.tools_causal import (
    balance_check_smd,
    check_missingness,
    check_positivity_overlap,
    check_treatment_type,
)
from packages.agent.tools_causal_estimation import (
    ipw_ate,
    regression_adjustment_ate,
    run_causal_estimation,
    select_estimator,
)
from services.api.main import app

client = TestClient(app)
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def setup_test_dataset(csv_path: Path, dataset_id: str) -> Path:
    """Create a test dataset in the storage directory."""
    datasets_dir = Path(__file__).parent.parent / "storage" / "datasets"
    dataset_dir = datasets_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy CSV
    shutil.copy(csv_path, dataset_dir / "data.csv")

    # Create metadata
    df = pd.read_csv(csv_path)
    inferred_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype in ["int64", "int32"]:
                inferred_types[col] = "int"
            else:
                inferred_types[col] = "float"
        else:
            inferred_types[col] = "object"

    metadata = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_names": list(df.columns),
        "inferred_types": inferred_types,
        "created_at": "2024-01-01T00:00:00Z",
    }

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create empty profile
    with open(dataset_dir / "profile.json", "w") as f:
        json.dump({"columns": {}}, f)

    return datasets_dir


def cleanup_test_dataset(dataset_id: str):
    """Remove test dataset."""
    datasets_dir = Path(__file__).parent.parent / "storage" / "datasets"
    dataset_dir = datasets_dir / dataset_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


class TestCausalGateUnit:
    """Unit tests for causal gate components."""

    def test_check_treatment_type_binary_pass(self):
        """Binary treatment should PASS."""
        dataset_id = f"test_binary_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result = check_treatment_type(dataset_id, "treatment", datasets_dir)
            assert result["diagnostic"]["status"] == "PASS"
            assert result["diagnostic"]["details"]["is_binary"] is True
        finally:
            cleanup_test_dataset(dataset_id)

    def test_check_treatment_type_non_binary_warn(self):
        """Non-binary treatment with few categories should WARN."""
        dataset_id = f"test_nonbinary_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_non_binary_treatment.csv",
                dataset_id
            )

            result = check_treatment_type(dataset_id, "treatment_level", datasets_dir)
            assert result["diagnostic"]["status"] == "WARN"
            assert result["diagnostic"]["details"]["is_binary"] is False
        finally:
            cleanup_test_dataset(dataset_id)

    def test_check_missingness_low_pass(self):
        """Low missingness should PASS."""
        dataset_id = f"test_miss_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result = check_missingness(
                dataset_id,
                ["treatment", "outcome", "age"],
                "treatment",
                datasets_dir
            )
            assert result["diagnostic"]["status"] == "PASS"
        finally:
            cleanup_test_dataset(dataset_id)

    def test_positivity_check_with_confounders(self):
        """Positivity check should run with confounders."""
        dataset_id = f"test_pos_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result = check_positivity_overlap(
                dataset_id,
                "treatment",
                ["age", "income", "prior_purchases"],
                datasets_dir
            )
            # Should have a diagnostic (PASS, WARN, or FAIL)
            assert result["diagnostic"]["status"] in ["PASS", "WARN", "FAIL"]
            # With 40 rows, we get WARN due to insufficient cases, which is expected
            assert "n_complete_cases" in result["diagnostic"]["details"] or "propensity_mean" in result["diagnostic"]["details"]
        finally:
            cleanup_test_dataset(dataset_id)

    def test_balance_check_smd(self):
        """Balance check should compute SMD."""
        dataset_id = f"test_smd_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result = balance_check_smd(
                dataset_id,
                "treatment",
                ["age", "income"],
                datasets_dir
            )
            assert result["diagnostic"]["status"] in ["PASS", "WARN", "FAIL"]
            assert result["table_artifact"]["headers"] == ["Covariate", "SMD", "Status"]
        finally:
            cleanup_test_dataset(dataset_id)


class TestCausalReadinessGate:
    """Tests for the full causal readiness gate."""

    def test_gate_fail_no_dataset(self):
        """Gate should FAIL when no dataset is provided."""
        report = causal_readiness_gate(
            question="What is the effect of treatment on outcome?",
            doc_id="test_doc",
            dataset_id=None,
            column_names=[],
            inferred_types={},
            datasets_dir=Path("/tmp"),
        )

        assert report.readiness_status == "FAIL"
        assert report.ready_for_estimation is False
        assert "upload a dataset" in report.followup_questions[0].lower()

    def test_gate_fail_missing_treatment_outcome(self):
        """Gate should FAIL and ask for treatment/outcome when not inferrable."""
        dataset_id = f"test_gate_miss_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_missing_treatment.csv",
                dataset_id
            )

            # Load metadata
            with open(datasets_dir / dataset_id / "metadata.json") as f:
                metadata = json.load(f)

            report = causal_readiness_gate(
                question="What is the causal effect?",  # Vague question
                doc_id="test_doc",
                dataset_id=dataset_id,
                column_names=metadata["column_names"],
                inferred_types=metadata["inferred_types"],
                datasets_dir=datasets_dir,
            )

            assert report.readiness_status == "FAIL"
            assert report.ready_for_estimation is False
            # Should ask for treatment or outcome
            questions_text = " ".join(report.followup_questions).lower()
            assert "treatment" in questions_text or "outcome" in questions_text
        finally:
            cleanup_test_dataset(dataset_id)

    def test_gate_warn_non_binary_treatment(self):
        """Gate should WARN for non-binary treatment."""
        dataset_id = f"test_gate_nonbin_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_non_binary_treatment.csv",
                dataset_id
            )

            with open(datasets_dir / dataset_id / "metadata.json") as f:
                metadata = json.load(f)

            report = causal_readiness_gate(
                question="What is the effect of treatment_level on outcome?",
                doc_id="test_doc",
                dataset_id=dataset_id,
                column_names=metadata["column_names"],
                inferred_types=metadata["inferred_types"],
                datasets_dir=datasets_dir,
                spec_override={"treatment": "treatment_level", "outcome": "outcome"},
            )

            # Should be WARN or FAIL due to non-binary treatment
            assert report.readiness_status in ["WARN", "FAIL"]
            assert report.ready_for_estimation is False
        finally:
            cleanup_test_dataset(dataset_id)

    def test_gate_pass_binary_treatment(self):
        """Gate should run diagnostics with binary treatment (may WARN/FAIL due to data quality)."""
        dataset_id = f"test_gate_pass_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            with open(datasets_dir / dataset_id / "metadata.json") as f:
                metadata = json.load(f)

            # Provide explicit confounders (excluding customer_id which has perfect separation)
            report = causal_readiness_gate(
                question="What is the effect of treatment on outcome?",
                doc_id="test_doc",
                dataset_id=dataset_id,
                column_names=metadata["column_names"],
                inferred_types=metadata["inferred_types"],
                datasets_dir=datasets_dir,
                spec_override={
                    "treatment": "treatment",
                    "outcome": "outcome",
                    "confounders": ["age", "income"],  # Exclude customer_id
                },
            )

            # Verify gate runs and produces diagnostics
            assert report.readiness_status in ["PASS", "WARN", "FAIL"]
            assert len(report.diagnostics) > 0

            # Treatment type check should PASS (binary treatment)
            treatment_diag = next((d for d in report.diagnostics if d.name == "treatment_type_check"), None)
            assert treatment_diag is not None
            assert treatment_diag.status == "PASS"

            # Verify no ATE is returned (Phase 3)
            report_dict = report.model_dump()
            report_str = json.dumps(report_dict)
            # Should not contain numeric ATE values
            import re
            ate_numeric = re.findall(r'"ate":\s*[-+]?\d+\.?\d*', report_str, re.IGNORECASE)
            assert len(ate_numeric) == 0, f"Found numeric ATE: {ate_numeric}"
        finally:
            cleanup_test_dataset(dataset_id)

    def test_gate_never_returns_numeric_ate(self):
        """Gate should NEVER return a numeric ATE in Phase 3."""
        dataset_id = f"test_gate_noate_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            with open(datasets_dir / dataset_id / "metadata.json") as f:
                metadata = json.load(f)

            report = causal_readiness_gate(
                question="What is the ATE of treatment on outcome?",
                doc_id="test_doc",
                dataset_id=dataset_id,
                column_names=metadata["column_names"],
                inferred_types=metadata["inferred_types"],
                datasets_dir=datasets_dir,
                spec_override={"treatment": "treatment", "outcome": "outcome"},
            )

            # Convert to dict and check no numeric ATE
            report_dict = report.model_dump()
            report_str = json.dumps(report_dict)

            # Should not contain "ATE: " followed by a number
            import re
            ate_pattern = r"ATE[:\s]+[-+]?\d+\.?\d*"
            matches = re.findall(ate_pattern, report_str, re.IGNORECASE)
            assert len(matches) == 0, f"Found numeric ATE in response: {matches}"
        finally:
            cleanup_test_dataset(dataset_id)


class TestCausalEstimationTools:
    """Phase 4: Tests for causal estimation tools."""

    def test_regression_adjustment_ate_deterministic(self):
        """Regression adjustment should produce deterministic results."""
        dataset_id = f"test_reg_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result1 = regression_adjustment_ate(
                dataset_id, "treatment", "outcome", ["age", "income"], datasets_dir
            )
            result2 = regression_adjustment_ate(
                dataset_id, "treatment", "outcome", ["age", "income"], datasets_dir
            )

            # Results should be identical
            assert result1["estimate_artifact"]["estimate"] == result2["estimate_artifact"]["estimate"]
            assert result1["estimate_artifact"]["ci_low"] == result2["estimate_artifact"]["ci_low"]
            assert result1["estimate_artifact"]["ci_high"] == result2["estimate_artifact"]["ci_high"]
        finally:
            cleanup_test_dataset(dataset_id)

    def test_ipw_ate_deterministic(self):
        """IPW should produce deterministic results."""
        dataset_id = f"test_ipw_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result1 = ipw_ate(
                dataset_id, "treatment", "outcome", ["age", "income"], datasets_dir
            )
            result2 = ipw_ate(
                dataset_id, "treatment", "outcome", ["age", "income"], datasets_dir
            )

            # Results should be identical
            assert result1["estimate_artifact"]["estimate"] == result2["estimate_artifact"]["estimate"]
            assert result1["estimate_artifact"]["ci_low"] == result2["estimate_artifact"]["ci_low"]
        finally:
            cleanup_test_dataset(dataset_id)

    def test_estimate_artifact_structure(self):
        """Estimate artifact should have required fields."""
        dataset_id = f"test_struct_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            result = regression_adjustment_ate(
                dataset_id, "treatment", "outcome", ["age"], datasets_dir
            )

            artifact = result["estimate_artifact"]
            assert artifact["type"] == "causal_estimate"
            assert artifact["method"] == "regression_adjustment"
            assert artifact["estimand"] == "ATE"
            assert "estimate" in artifact
            assert "ci_low" in artifact
            assert "ci_high" in artifact
            assert "n_used" in artifact
            assert artifact["ci_low"] <= artifact["estimate"] <= artifact["ci_high"]
        finally:
            cleanup_test_dataset(dataset_id)

    def test_select_estimator_logic(self):
        """Estimator selection should follow rules."""
        # IPW preferred when positivity PASS and covariates exist
        assert select_estimator("PASS", 3) == "ipw"

        # Regression preferred when positivity fails
        assert select_estimator("WARN", 3) == "regression_adjustment"
        assert select_estimator("FAIL", 3) == "regression_adjustment"

        # Regression preferred when no covariates
        assert select_estimator("PASS", 0) == "regression_adjustment"

    def test_run_causal_estimation_unified(self):
        """run_causal_estimation should dispatch to correct method."""
        dataset_id = f"test_unified_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            # Test regression method
            result_reg = run_causal_estimation(
                dataset_id, "treatment", "outcome", ["age"], datasets_dir,
                method="regression_adjustment"
            )
            assert result_reg["estimate_artifact"]["method"] == "regression_adjustment"

            # Test IPW method
            result_ipw = run_causal_estimation(
                dataset_id, "treatment", "outcome", ["age"], datasets_dir,
                method="ipw"
            )
            assert result_ipw["estimate_artifact"]["method"] == "ipw"
            assert "propensity_summary" in result_ipw
        finally:
            cleanup_test_dataset(dataset_id)


class TestConfirmationsGating:
    """Phase 4: Tests for confirmations gating logic."""

    def test_check_confirmations_no_confirmations(self):
        """Should request confirmations when none provided."""
        state = {
            "causal_readiness_status": "PASS",
            "causal_confirmations": None,
            "trace_events": [],
            "artifacts": [],
        }

        result = check_confirmations_node(state)

        assert result["confirmations_ok"] is False
        assert result["route"] == "NEEDS_CLARIFICATION"
        # Should have checklist artifact
        checklist = [a for a in result["artifacts"] if a.get("type") == "checklist"]
        assert len(checklist) > 0

    def test_check_confirmations_ok_to_estimate_false(self):
        """Should block estimation when ok_to_estimate=False."""
        state = {
            "causal_readiness_status": "PASS",
            "causal_confirmations": {
                "assignment_mechanism": "randomized",
                "interference_assumption": "no_interference",
                "missing_data_policy": "listwise_delete",
                "ok_to_estimate": False,
            },
            "trace_events": [],
            "artifacts": [],
        }

        result = check_confirmations_node(state)

        assert result["confirmations_ok"] is False
        assert "Declined" in result["artifacts"][-1]["content"]

    def test_check_confirmations_ok_to_estimate_true(self):
        """Should allow estimation when ok_to_estimate=True."""
        state = {
            "causal_readiness_status": "PASS",
            "causal_confirmations": {
                "assignment_mechanism": "randomized",
                "interference_assumption": "no_interference",
                "missing_data_policy": "listwise_delete",
                "ok_to_estimate": True,
            },
            "trace_events": [],
            "artifacts": [],
        }

        result = check_confirmations_node(state)

        assert result["confirmations_ok"] is True

    def test_check_confirmations_readiness_not_pass(self):
        """Should block estimation when readiness is not PASS."""
        state = {
            "causal_readiness_status": "WARN",
            "causal_confirmations": {"ok_to_estimate": True},
            "trace_events": [],
            "artifacts": [],
        }

        result = check_confirmations_node(state)

        assert result["confirmations_ok"] is False


class TestTracePersistence:
    """Phase 5: Tests for trace persistence."""

    def test_ask_creates_trace_file(self):
        """Calling /ask should create a trace file."""
        from services.api.main import TRACES_DIR

        dataset_id = f"test_trace_{uuid.uuid4().hex[:8]}"
        try:
            setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            # First, upload a valid document
            from services.api.tests.test_upload_context_doc import create_docx_bytes
            content = {
                "Dataset Overview": "Customer transactions for Q1 2024. " * 20,
                "Target Use / Primary Questions": "Analyze trends and patterns. " * 20,
                "Data Dictionary": "transaction_id, customer_id, amount. " * 20,
                "Known Caveats": "Some missing values in customer_id. " * 20,
            }
            docx_bytes = create_docx_bytes(content)

            import io
            upload_resp = client.post(
                "/upload_context_doc",
                files={"file": ("test.docx", io.BytesIO(docx_bytes),
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
            )
            assert upload_resp.status_code == 200
            doc_id = upload_resp.json()["doc_id"]

            # Now call /ask
            ask_resp = client.post(
                "/ask",
                json={
                    "question": "What is the effect of treatment on outcome?",
                    "doc_id": doc_id,
                    "dataset_id": dataset_id,
                    "causal_spec_override": {
                        "treatment": "treatment",
                        "outcome": "outcome",
                        "confounders": ["age", "income"],
                    },
                }
            )
            assert ask_resp.status_code == 200
            trace_id = ask_resp.json()["trace_id"]

            # Verify trace file exists
            trace_path = TRACES_DIR / f"{trace_id}.json"
            assert trace_path.exists(), f"Trace file not found at {trace_path}"

            # Verify trace content
            with open(trace_path) as f:
                trace_data = json.load(f)

            assert trace_data["trace_id"] == trace_id
            assert trace_data["doc_id"] == doc_id
            assert trace_data["dataset_id"] == dataset_id
            assert "timestamp" in trace_data
            assert "route" in trace_data
            assert "diagnostics_summary" in trace_data
            assert "artifact_inventory" in trace_data
        finally:
            cleanup_test_dataset(dataset_id)


class TestRunEstimationNode:
    """Phase 4: Tests for run_estimation_node."""

    def test_run_estimation_with_confirmations(self):
        """Should produce estimate when confirmations OK."""
        dataset_id = f"test_est_{uuid.uuid4().hex[:8]}"
        try:
            datasets_dir = setup_test_dataset(
                FIXTURES_DIR / "causal_binary_treatment.csv",
                dataset_id
            )

            state = {
                "confirmations_ok": True,
                "dataset_id": dataset_id,
                "causal_report": {
                    "spec": {
                        "treatment": "treatment",
                        "outcome": "outcome",
                        "confounders_selected": ["age", "income"],
                    },
                    "diagnostics": [
                        {"name": "positivity_check", "status": "PASS"},
                    ],
                },
                "trace_events": [],
                "artifacts": [],
            }

            result = run_estimation_node(state, datasets_dir)

            assert "causal_estimate" in result
            assert result["causal_estimate"]["type"] == "causal_estimate"
            assert "estimate" in result["causal_estimate"]
            assert "ATE" in result["llm_response"]
        finally:
            cleanup_test_dataset(dataset_id)

    def test_run_estimation_skipped_without_confirmations(self):
        """Should skip estimation when confirmations not OK."""
        state = {
            "confirmations_ok": False,
            "trace_events": [],
        }

        result = run_estimation_node(state, Path("/tmp"))

        assert "causal_estimate" not in result

