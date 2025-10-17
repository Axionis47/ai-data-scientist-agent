"""Unit tests for botds.tools.artifacts module."""

import pytest
from pathlib import Path

from botds.tools.artifacts import ArtifactStore, HandoffLedger


@pytest.fixture
def artifact_store(tmp_path):
    """Create ArtifactStore instance."""
    return ArtifactStore(artifacts_dir=tmp_path)


@pytest.fixture
def handoff_ledger(tmp_path):
    """Create HandoffLedger instance."""
    return HandoffLedger(artifacts_dir=tmp_path)


class TestArtifactStore:
    """Tests for ArtifactStore class."""
    
    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        store = ArtifactStore(artifacts_dir=tmp_path)
        
        assert store.artifacts_dir == tmp_path
        assert store.reports_dir.exists()
        assert store.tables_dir.exists()
        assert store.figures_dir.exists()
    
    def test_write_report_one_pager_html(self, artifact_store):
        """Test writing one-pager HTML report."""
        payload = {
            "business_goal": "Predict customer churn",
            "primary_metric": "F1 Score",
            "dataset_info": {"shape": "1000 rows × 20 columns"},
            "target": "churn",
            "best_model": "Random Forest",
            "best_score": 0.85
        }
        
        result = artifact_store.write_report(
            kind="one_pager",
            payload=payload,
            format="html"
        )
        
        assert "report_ref" in result
        assert result["kind"] == "one_pager"
        assert result["format"] == "html"
        assert "size_bytes" in result
        
        # Check file exists
        report_path = Path(result["report_ref"])
        assert report_path.exists()
        assert report_path.suffix == ".html"
    
    def test_write_report_one_pager_markdown(self, artifact_store):
        """Test writing one-pager Markdown report."""
        payload = {
            "business_goal": "Predict customer churn",
            "primary_metric": "F1 Score"
        }
        
        result = artifact_store.write_report(
            kind="one_pager",
            payload=payload,
            format="md"
        )
        
        assert result["format"] == "md"
        report_path = Path(result["report_ref"])
        assert report_path.suffix == ".md"
    
    def test_write_report_appendix_html(self, artifact_store):
        """Test writing appendix HTML report."""
        payload = {
            "data_profile": {"columns": 20, "rows": 1000},
            "model_details": {"type": "RandomForest", "params": {}}
        }
        
        result = artifact_store.write_report(
            kind="appendix",
            payload=payload,
            format="html"
        )
        
        assert result["kind"] == "appendix"
        assert result["format"] == "html"
        
        report_path = Path(result["report_ref"])
        assert report_path.exists()
    
    def test_write_report_appendix_markdown(self, artifact_store):
        """Test writing appendix Markdown report."""
        payload = {
            "data_profile": {"columns": 20}
        }
        
        result = artifact_store.write_report(
            kind="appendix",
            payload=payload,
            format="md"
        )
        
        assert result["kind"] == "appendix"
        assert result["format"] == "md"
    
    def test_write_report_invalid_kind(self, artifact_store):
        """Test that invalid report kind raises error."""
        with pytest.raises(ValueError, match="Unknown report kind"):
            artifact_store.write_report(
                kind="invalid_kind",
                payload={},
                format="html"
            )
    
    def test_write_report_contains_content(self, artifact_store):
        """Test that written report contains expected content."""
        payload = {
            "business_goal": "Test Goal",
            "primary_metric": "Accuracy"
        }
        
        result = artifact_store.write_report(
            kind="one_pager",
            payload=payload,
            format="html"
        )
        
        # Read the file and check content
        with open(result["report_ref"], "r") as f:
            content = f.read()
        
        assert "Test Goal" in content
        assert "Accuracy" in content
    
    def test_write_report_size_bytes(self, artifact_store):
        """Test that size_bytes is reported."""
        payload = {"business_goal": "Test"}

        result = artifact_store.write_report(
            kind="one_pager",
            payload=payload,
            format="html"
        )

        # Check that size_bytes is present and reasonable
        assert "size_bytes" in result
        assert result["size_bytes"] > 0
    
    def test_get_function_definitions(self, artifact_store):
        """Test that function definitions are returned."""
        defs = artifact_store.get_function_definitions()
        
        assert isinstance(defs, list)
        assert len(defs) > 0
        
        # Check structure
        func = defs[0]["function"]
        assert func["name"] == "ArtifactStore_write_report"
        assert "description" in func
        assert "parameters" in func


class TestHandoffLedger:
    """Tests for HandoffLedger class."""

    def test_init_creates_ledger(self, tmp_path):
        """Test that initialization creates ledger."""
        ledger = HandoffLedger(artifacts_dir=tmp_path)

        assert ledger.ledger is not None

    def test_append_handoff(self, handoff_ledger):
        """Test appending a handoff."""
        result = handoff_ledger.append(
            job_id="test_job_123",
            stage="profiling",
            input_refs=["data.csv"],
            output_refs=["profile.json", "data.pkl"],
            schema_uri="schema://v1",
            hash_value="abc123"
        )

        assert result["status"] == "logged"
        assert result["stage"] == "profiling"
        assert result["outputs"] == 2

    def test_append_multiple_handoffs(self, handoff_ledger):
        """Test appending multiple handoffs."""
        result1 = handoff_ledger.append(
            job_id="job1",
            stage="stage1",
            input_refs=["file1.csv"],
            output_refs=["file1.pkl"],
            schema_uri="schema://v1",
            hash_value="hash1"
        )

        result2 = handoff_ledger.append(
            job_id="job2",
            stage="stage2",
            input_refs=["file1.pkl"],
            output_refs=["file2.pkl"],
            schema_uri="schema://v1",
            hash_value="hash2"
        )

        assert result1["status"] == "logged"
        assert result2["status"] == "logged"

    def test_append_with_empty_refs(self, handoff_ledger):
        """Test appending with empty references."""
        result = handoff_ledger.append(
            job_id="job1",
            stage="stage1",
            input_refs=[],
            output_refs=[],
            schema_uri="schema://v1",
            hash_value="hash1"
        )

        assert result["status"] == "logged"
        assert result["outputs"] == 0

    def test_get_function_definitions(self, handoff_ledger):
        """Test that function definitions are returned."""
        defs = handoff_ledger.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure
        func = defs[0]["function"]
        assert func["name"] == "HandoffLedger_append"
        assert "description" in func
        assert "parameters" in func

