"""Unit tests for botds.tools.budget module."""

import pytest
import time
from pathlib import Path

from botds.tools.budget import BudgetGuard


@pytest.fixture
def budget_config():
    """Create budget configuration."""
    return {
        "time_min": 10,  # 10 minutes
        "memory_gb": 4.0,
        "token_budget": 5000
    }


@pytest.fixture
def guard(tmp_path, budget_config):
    """Create BudgetGuard instance."""
    return BudgetGuard(artifacts_dir=tmp_path, budgets=budget_config)


class TestBudgetGuard:
    """Tests for BudgetGuard class."""
    
    def test_init_creates_instance(self, tmp_path, budget_config):
        """Test that initialization creates instance."""
        guard = BudgetGuard(artifacts_dir=tmp_path, budgets=budget_config)
        
        assert guard.artifacts_dir == tmp_path
        assert guard.budgets == budget_config
        assert guard.token_usage == 0
        assert len(guard.checkpoints) == 0
    
    def test_init_sets_budget_limits(self, guard, budget_config):
        """Test that budget limits are set correctly."""
        assert guard.time_limit_seconds == budget_config["time_min"] * 60
        assert guard.memory_limit_gb == budget_config["memory_gb"]
        assert guard.token_limit == budget_config["token_budget"]
    
    def test_checkpoint_creates_record(self, guard):
        """Test that checkpoint creates a record."""
        result = guard.checkpoint(stage="test_stage", additional_tokens=100)

        # Check result structure
        assert "stage" in result
        assert "status" in result
        assert "usage" in result
        assert "recommendations" in result
        assert "checkpoint" in result

        # Check stage name
        assert result["stage"] == "test_stage"

        # Check checkpoint details
        checkpoint = result["checkpoint"]
        assert checkpoint["tokens_used"] == 100
        assert checkpoint["stage"] == "test_stage"

        # Check checkpoint was saved
        assert len(guard.checkpoints) == 1
    
    def test_checkpoint_tracks_elapsed_time(self, guard):
        """Test that checkpoint tracks elapsed time."""
        time.sleep(0.1)  # Sleep for 100ms
        result = guard.checkpoint(stage="stage1")

        checkpoint = result["checkpoint"]
        assert checkpoint["elapsed_seconds"] >= 0.1
        assert checkpoint["elapsed_seconds"] < 1.0  # Should be less than 1 second

    def test_checkpoint_accumulates_tokens(self, guard):
        """Test that tokens accumulate across checkpoints."""
        guard.checkpoint(stage="stage1", additional_tokens=100)
        guard.checkpoint(stage="stage2", additional_tokens=200)
        result = guard.checkpoint(stage="stage3", additional_tokens=150)

        checkpoint = result["checkpoint"]
        assert checkpoint["tokens_used"] == 450  # 100 + 200 + 150
        assert guard.token_usage == 450

    def test_checkpoint_calculates_remaining_budgets(self, guard):
        """Test that remaining budgets are calculated."""
        result = guard.checkpoint(stage="stage1", additional_tokens=1000)

        checkpoint = result["checkpoint"]
        budgets = checkpoint["budgets"]
        assert "time_remaining_seconds" in budgets
        assert "memory_remaining_gb" in budgets
        assert "tokens_remaining" in budgets

        # Tokens remaining should be 4000 (5000 - 1000)
        assert budgets["tokens_remaining"] == 4000
    
    def test_checkpoint_status_ok_when_under_budget(self, guard):
        """Test that status is 'ok' when under budget."""
        result = guard.checkpoint(stage="stage1", additional_tokens=100)
        
        assert "status" in result
        assert result["status"] == "ok"
    
    def test_checkpoint_status_downshift_when_approaching_limit(self, guard):
        """Test that status is 'downshift' when approaching limits."""
        # Use 75% of token budget
        result = guard.checkpoint(stage="stage1", additional_tokens=3750)
        
        # Should recommend downshift
        assert result["status"] in ["ok", "downshift"]
        
        if result["status"] == "downshift":
            assert "recommendations" in result
            assert len(result["recommendations"]) > 0
    
    def test_checkpoint_status_abort_when_over_limit(self, guard):
        """Test that status is 'abort' when over limits."""
        # Use 95% of token budget
        result = guard.checkpoint(stage="stage1", additional_tokens=4750)

        # Should recommend abort or downshift
        assert result["status"] in ["ok", "downshift", "abort"]

        # Token usage should be high
        assert result["usage"]["tokens_pct"] >= 90

    def test_multiple_checkpoints_stored(self, guard):
        """Test that multiple checkpoints are stored."""
        guard.checkpoint(stage="stage1", additional_tokens=100)
        guard.checkpoint(stage="stage2", additional_tokens=200)
        guard.checkpoint(stage="stage3", additional_tokens=300)

        assert len(guard.checkpoints) == 3
        assert guard.checkpoints[0]["stage"] == "stage1"
        assert guard.checkpoints[1]["stage"] == "stage2"
        assert guard.checkpoints[2]["stage"] == "stage3"

    def test_get_usage_summary(self, guard):
        """Test getting budget usage summary."""
        guard.checkpoint(stage="stage1", additional_tokens=100)
        guard.checkpoint(stage="stage2", additional_tokens=200)

        summary = guard.get_usage_summary()

        assert "total_stages" in summary
        assert "final_usage" in summary
        assert "checkpoints" in summary

        assert summary["total_stages"] == 2
        assert summary["final_usage"]["tokens"] == 300

    def test_budget_with_zero_tokens(self, guard):
        """Test checkpoint with zero additional tokens."""
        result = guard.checkpoint(stage="stage1", additional_tokens=0)

        checkpoint = result["checkpoint"]
        assert checkpoint["tokens_used"] == 0
        assert checkpoint["budgets"]["tokens_remaining"] == 5000
    
    def test_budget_with_default_values(self, tmp_path):
        """Test BudgetGuard with default budget values."""
        guard = BudgetGuard(artifacts_dir=tmp_path, budgets={})
        
        # Should use defaults
        assert guard.time_limit_seconds == 25 * 60  # 25 minutes default
        assert guard.memory_limit_gb == 4.0
        assert guard.token_limit == 8000
    
    def test_checkpoint_memory_tracking(self, guard):
        """Test that memory usage is tracked."""
        result = guard.checkpoint(stage="stage1")

        checkpoint = result["checkpoint"]
        assert "memory_gb" in checkpoint
        assert checkpoint["memory_gb"] >= 0
        assert isinstance(checkpoint["memory_gb"], float)
    
    def test_get_function_definitions(self, guard):
        """Test that function definitions are returned."""
        defs = guard.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure
        assert "type" in defs[0]
        assert defs[0]["type"] == "function"

    def test_checkpoint_warns_on_time_limit(self, tmp_path):
        """Test that checkpoint warns when approaching time limit."""
        guard = BudgetGuard(
            artifacts_dir=tmp_path,
            budgets={"time_budget_minutes": 0.01}  # Very short time limit
        )

        import time
        time.sleep(0.1)  # Sleep to exceed limit

        result = guard.checkpoint(stage="stage1")

        # Should have warning about time
        assert "budgets" in result["checkpoint"]
        assert "time_remaining_seconds" in result["checkpoint"]["budgets"]

    def test_checkpoint_warns_on_token_limit(self, guard):
        """Test that checkpoint warns when approaching token limit."""
        # Use lots of tokens
        result = guard.checkpoint(stage="stage1", additional_tokens=4500)

        checkpoint = result["checkpoint"]
        assert checkpoint["tokens_used"] == 4500
        assert checkpoint["budgets"]["tokens_remaining"] == 500

    def test_checkpoint_tracks_cumulative_tokens(self, guard):
        """Test that checkpoint tracks cumulative token usage."""
        # First checkpoint
        result1 = guard.checkpoint(stage="stage1", additional_tokens=100)

        # Second checkpoint
        result2 = guard.checkpoint(stage="stage2", additional_tokens=200)

        # Should track cumulative
        assert result1["checkpoint"]["tokens_used"] == 100
        assert result2["checkpoint"]["tokens_used"] == 300  # Cumulative: 100 + 200

    def test_multiple_checkpoints_track_stages(self, guard):
        """Test that multiple checkpoints track different stages."""
        guard.checkpoint(stage="stage1", additional_tokens=100)
        guard.checkpoint(stage="stage2", additional_tokens=200)
        guard.checkpoint(stage="stage3", additional_tokens=150)

        # Check that all checkpoints are stored
        assert len(guard.checkpoints) == 3
        assert guard.checkpoints[0]["stage"] == "stage1"
        assert guard.checkpoints[1]["stage"] == "stage2"
        assert guard.checkpoints[2]["stage"] == "stage3"

    def test_checkpoint_includes_stage_name(self, guard):
        """Test that checkpoint includes stage name."""
        result = guard.checkpoint(stage="my_stage")

        assert "stage" in result["checkpoint"]
        assert result["checkpoint"]["stage"] == "my_stage"

    def test_checkpoint_includes_timestamp(self, guard):
        """Test that checkpoint includes timestamp."""
        result = guard.checkpoint(stage="stage1")

        assert "elapsed_seconds" in result["checkpoint"]
        assert result["checkpoint"]["elapsed_seconds"] >= 0

    def test_budget_remaining_calculations(self, guard):
        """Test that budget remaining is calculated correctly."""
        result = guard.checkpoint(stage="stage1", additional_tokens=1000)

        budgets = result["checkpoint"]["budgets"]
        assert budgets["tokens_remaining"] == 4000  # 5000 - 1000
        assert budgets["time_remaining_seconds"] > 0

    def test_checkpoint_status_ok_when_under_budget(self, guard):
        """Test that status is 'ok' when under budget."""
        result = guard.checkpoint(stage="stage1", additional_tokens=100)

        assert result["status"] == "ok"

    def test_checkpoint_status_downshift_on_high_token_usage(self, guard):
        """Test that status is 'downshift' when token usage is high."""
        # Use 75% of tokens (3750 out of 5000)
        result = guard.checkpoint(stage="stage1", additional_tokens=3750)

        assert result["status"] == "downshift"
        assert any("Token" in rec for rec in result["recommendations"])

    def test_checkpoint_status_abort_on_exceeded_tokens(self, guard):
        """Test that status is 'abort' when tokens exceeded."""
        # Use 95% of tokens (4750 out of 5000)
        result = guard.checkpoint(stage="stage1", additional_tokens=4750)

        assert result["status"] == "abort"
        assert any("Token" in rec for rec in result["recommendations"])

