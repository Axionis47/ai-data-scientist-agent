"""Unit tests for botds.pipeline module."""

import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from botds.pipeline import Pipeline
from botds.config import Config
from botds.utils import save_pickle


@pytest.fixture
def minimal_config(tmp_path):
    """Create minimal config for testing."""
    config_dict = {
        "data": {
            "source": "builtin",
            "name": "iris",
            "target": "species"
        },
        "cache": {
            "dir": str(tmp_path / "cache"),
            "mode": "cold"
        },
        "report": {
            "out_dir": str(tmp_path / "reports")
        },
        "llms": {
            "default_model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "budgets": {
            "max_tokens": 100000,
            "max_cost_usd": 10.0,
            "max_time_minutes": 60
        },
        "split": {
            "policy": "iid",
            "train_frac": 0.7,
            "val_frac": 0.15,
            "test_frac": 0.15
        },
        "pii": {
            "enabled": False,
            "patterns": [],
            "action": "redact"
        }
    }
    return Config(**config_dict)


@pytest.fixture
def pipeline(minimal_config):
    """Create Pipeline instance."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        return Pipeline(config=minimal_config)


class TestPipelineInitialization:
    """Tests for Pipeline initialization."""

    def test_init_creates_directories(self, minimal_config):
        """Test that initialization creates necessary directories."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.artifacts_dir.exists()
            assert pipeline.handoffs_dir.exists()
            assert pipeline.logs_dir.exists()

    def test_init_sets_job_id(self, minimal_config):
        """Test that job ID is generated."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.job_id is not None
            assert len(pipeline.job_id) > 0

    def test_init_creates_cache(self, minimal_config):
        """Test that cache is initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.cache is not None
            assert pipeline.cache.mode == "cold"

    def test_init_creates_decision_log(self, minimal_config):
        """Test that decision log is initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.decision_log is not None
            assert pipeline.decision_log.log_path.parent.exists()

    def test_init_creates_handoff_ledger(self, minimal_config):
        """Test that handoff ledger is initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.handoff_ledger is not None

    def test_init_creates_manifest(self, minimal_config):
        """Test that run manifest is initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.manifest is not None
            assert pipeline.manifest.job_id == pipeline.job_id

    def test_init_creates_llm_router(self, minimal_config):
        """Test that LLM router is initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.llm_router is not None

    def test_init_creates_tools(self, minimal_config):
        """Test that all tools are initialized."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            expected_tools = [
                "data_store", "schema_profiler", "quality_guard",
                "splitter", "featurizer", "model_trainer", "tuner",
                "metrics", "calibrator", "fairness", "robustness",
                "plotter", "artifact_store", "handoff_ledger",
                "budget_guard", "pii"
            ]

            for tool_name in expected_tools:
                assert tool_name in pipeline.tools
                assert pipeline.tools[tool_name] is not None

    def test_init_sets_current_stage(self, minimal_config):
        """Test that current stage is set to initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.current_stage == "initialization"

    def test_init_creates_empty_stage_outputs(self, minimal_config):
        """Test that stage outputs dict is created."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline = Pipeline(config=minimal_config)

            assert pipeline.stage_outputs == {}

    def test_different_pipelines_different_job_ids(self, minimal_config):
        """Test that different pipeline instances get different job IDs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            pipeline1 = Pipeline(config=minimal_config)
            pipeline2 = Pipeline(config=minimal_config)

            assert pipeline1.job_id != pipeline2.job_id


class TestPipelineStageManagement:
    """Tests for pipeline stage management."""
    
    def test_run_stage_updates_current_stage(self, pipeline):
        """Test that _run_stage updates current_stage."""
        def mock_stage():
            return {"result": "success"}
        
        with patch.object(pipeline.tools["budget_guard"], "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {
                "status": "ok",
                "checkpoint": {
                    "stage": "test_stage",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "elapsed_seconds": 0.0,
                    "memory_gb": 0.1,
                    "tokens_used": 0,
                    "budgets": {}
                }
            }
            
            pipeline._run_stage("test_stage", mock_stage)
            
            assert pipeline.current_stage == "test_stage"
    
    def test_run_stage_stores_output(self, pipeline):
        """Test that _run_stage stores stage output."""
        def mock_stage():
            return {"result": "success", "data": "test"}
        
        with patch.object(pipeline.tools["budget_guard"], "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {
                "status": "ok",
                "checkpoint": {
                    "stage": "test_stage",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "elapsed_seconds": 0.0,
                    "memory_gb": 0.1,
                    "tokens_used": 0,
                    "budgets": {}
                }
            }
            
            pipeline._run_stage("test_stage", mock_stage)
            
            assert "test_stage" in pipeline.stage_outputs
            assert pipeline.stage_outputs["test_stage"]["result"] == "success"
            assert pipeline.stage_outputs["test_stage"]["data"] == "test"
    
    def test_run_stage_checks_budget(self, pipeline):
        """Test that _run_stage checks budget before running."""
        def mock_stage():
            return {"result": "success"}
        
        with patch.object(pipeline.tools["budget_guard"], "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {
                "status": "ok",
                "checkpoint": {
                    "stage": "test_stage",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "elapsed_seconds": 0.0,
                    "memory_gb": 0.1,
                    "tokens_used": 0,
                    "budgets": {}
                }
            }
            
            pipeline._run_stage("test_stage", mock_stage)
            
            mock_checkpoint.assert_called_once_with("test_stage")
    
    def test_run_stage_aborts_on_budget_exceeded(self, pipeline):
        """Test that _run_stage aborts if budget is exceeded."""
        def mock_stage():
            return {"result": "success"}
        
        with patch.object(pipeline.tools["budget_guard"], "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {
                "status": "abort",
                "checkpoint": {
                    "stage": "test_stage",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "elapsed_seconds": 0.0,
                    "memory_gb": 0.1,
                    "tokens_used": 100000,
                    "budgets": {}
                },
                "recommendations": "Token budget exceeded"
            }
            
            with pytest.raises(RuntimeError, match="Budget exceeded"):
                pipeline._run_stage("test_stage", mock_stage)
    
    def test_run_stage_updates_manifest(self, pipeline):
        """Test that _run_stage updates the manifest."""
        def mock_stage():
            return {"result": "success"}
        
        with patch.object(pipeline.tools["budget_guard"], "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {
                "status": "ok",
                "checkpoint": {
                    "stage": "test_stage",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "elapsed_seconds": 0.0,
                    "memory_gb": 0.1,
                    "tokens_used": 0,
                    "budgets": {}
                }
            }
            
            with patch.object(pipeline.manifest, "add_budget_usage") as mock_add_budget:
                pipeline._run_stage("test_stage", mock_stage)
                
                mock_add_budget.assert_called_once()
                call_args = mock_add_budget.call_args
                assert call_args[0][0] == "test_stage"
                assert "duration_seconds" in call_args[0][1]
                assert "budget_status" in call_args[0][1]


class TestPipelineRun:
    """Tests for pipeline run method."""
    
    def test_run_returns_success_dict(self, pipeline):
        """Test that run returns success dict on completion."""
        # Mock all stage methods
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            with patch.object(pipeline.manifest, "save") as mock_save:
                result = pipeline.run()
                
                assert result["status"] == "success"
                assert result["job_id"] == pipeline.job_id
                assert "artifacts_dir" in result
                assert "cache_stats" in result
    
    def test_run_executes_all_stages(self, pipeline):
        """Test that run executes all 7 stages."""
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            with patch.object(pipeline.manifest, "save") as mock_save:
                pipeline.run()
                
                # Should call _run_stage 7 times (one for each stage)
                assert mock_run_stage.call_count == 7
                
                # Check stage names
                stage_names = [call[0][0] for call in mock_run_stage.call_args_list]
                expected_stages = [
                    "intake_validation",
                    "profiling_quality",
                    "eda_hypotheses",
                    "feature_splits",
                    "model_ladder",
                    "evaluation_stress",
                    "reporting"
                ]
                assert stage_names == expected_stages
    
    def test_run_saves_manifest(self, pipeline):
        """Test that run saves the manifest."""
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            with patch.object(pipeline.manifest, "save") as mock_save:
                pipeline.run()
                
                mock_save.assert_called_once()
                call_args = mock_save.call_args[0][0]
                assert str(call_args).endswith("run_manifest.json")
    
    def test_run_returns_failure_on_exception(self, pipeline):
        """Test that run returns failure dict on exception."""
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            mock_run_stage.side_effect = Exception("Test error")

            result = pipeline.run()

            assert result["status"] == "failed"
            assert result["job_id"] == pipeline.job_id
            assert "error" in result
            assert "Test error" in result["error"]

    def test_run_includes_artifacts_dir(self, pipeline):
        """Test that run result includes artifacts directory."""
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            with patch.object(pipeline.manifest, "save") as mock_save:
                result = pipeline.run()

                assert "artifacts_dir" in result
                assert str(pipeline.artifacts_dir) in result["artifacts_dir"]

    def test_run_includes_job_id(self, pipeline):
        """Test that run result includes job ID."""
        with patch.object(pipeline, "_run_stage") as mock_run_stage:
            with patch.object(pipeline.manifest, "save") as mock_save:
                result = pipeline.run()

                assert "job_id" in result
                assert result["job_id"] == pipeline.job_id

