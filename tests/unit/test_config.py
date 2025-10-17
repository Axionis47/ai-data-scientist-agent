"""Unit tests for botds.config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from botds.config import (
    BudgetConfig,
    CacheConfig,
    Config,
    DataConfig,
    FairnessConfig,
    LLMConfig,
    PIIConfig,
    ReportConfig,
    SamplingConfig,
    SplitConfig,
)


class TestDataConfig:
    """Test DataConfig model."""

    def test_default_builtin_config(self):
        """Test default builtin data configuration."""
        config = DataConfig()
        assert config.source == "builtin"
        assert config.name == "iris"
        assert config.csv_paths == []
        assert config.target == ""

    def test_csv_config_valid(self):
        """Test valid CSV configuration."""
        config = DataConfig(
            source="csv",
            name="my_data",
            csv_paths=["/path/to/data.csv"],
            target="target_col"
        )
        assert config.source == "csv"
        assert config.name == "my_data"
        assert len(config.csv_paths) == 1
        assert config.target == "target_col"

    def test_builtin_datasets(self):
        """Test builtin dataset names."""
        for dataset in ["iris", "breast_cancer", "diabetes"]:
            config = DataConfig(source="builtin", name=dataset)
            assert config.name == dataset


class TestBudgetConfig:
    """Test BudgetConfig model."""

    def test_default_budget(self):
        """Test default budget configuration."""
        config = BudgetConfig()
        assert config.time_min == 25
        assert config.memory_gb == 4.0
        assert config.token_budget == 8000

    def test_custom_budget(self):
        """Test custom budget configuration."""
        config = BudgetConfig(
            time_min=10,
            memory_gb=2.0,
            token_budget=5000
        )
        assert config.time_min == 10
        assert config.memory_gb == 2.0
        assert config.token_budget == 5000


class TestFairnessConfig:
    """Test FairnessConfig model."""

    def test_default_fairness(self):
        """Test default fairness configuration."""
        config = FairnessConfig()
        assert config.enabled is False
        assert config.sensitive_cols == []
        assert config.policy == "report"

    def test_enabled_fairness(self):
        """Test enabled fairness configuration."""
        config = FairnessConfig(
            enabled=True,
            sensitive_cols=["gender", "race"],
            policy="block"
        )
        assert config.enabled is True
        assert len(config.sensitive_cols) == 2
        assert config.policy == "block"


class TestPIIConfig:
    """Test PIIConfig model."""

    def test_default_pii(self):
        """Test default PII configuration."""
        config = PIIConfig()
        assert config.enabled is True
        assert "email" in config.patterns
        assert "phone" in config.patterns
        assert config.action == "redact"

    def test_custom_pii(self):
        """Test custom PII configuration."""
        config = PIIConfig(
            enabled=False,
            patterns=["ssn", "credit_card"],
            action="block"
        )
        assert config.enabled is False
        assert "ssn" in config.patterns
        assert config.action == "block"


class TestSplitConfig:
    """Test SplitConfig model."""

    def test_default_split(self):
        """Test default split configuration."""
        config = SplitConfig()
        assert config.policy == "iid"
        assert config.time_col == ""
        assert config.test_size == 0.2
        assert config.val_size == 0.2
        assert config.rolling is False
        assert config.seed == 42

    def test_time_split(self):
        """Test time-based split configuration."""
        config = SplitConfig(
            policy="time",
            time_col="timestamp",
            rolling=True
        )
        assert config.policy == "time"
        assert config.time_col == "timestamp"
        assert config.rolling is True

    def test_custom_split_sizes(self):
        """Test custom split sizes."""
        config = SplitConfig(
            test_size=0.3,
            val_size=0.15,
            seed=123
        )
        assert config.test_size == 0.3
        assert config.val_size == 0.15
        assert config.seed == 123


class TestSamplingConfig:
    """Test SamplingConfig model."""

    def test_default_sampling(self):
        """Test default sampling configuration."""
        config = SamplingConfig()
        assert config.eda_rows == 200000
        assert config.stratify_by == []

    def test_custom_sampling(self):
        """Test custom sampling configuration."""
        config = SamplingConfig(
            eda_rows=100000,
            stratify_by=["category"]
        )
        assert config.eda_rows == 100000
        assert "category" in config.stratify_by


class TestCacheConfig:
    """Test CacheConfig model."""

    def test_default_cache(self):
        """Test default cache configuration."""
        config = CacheConfig()
        assert config.mode == "warm"
        assert config.dir == "./cache"

    def test_cache_modes(self):
        """Test different cache modes."""
        for mode in ["warm", "cold", "paranoid"]:
            config = CacheConfig(mode=mode)
            assert config.mode == mode


class TestReportConfig:
    """Test ReportConfig model."""

    def test_default_report(self):
        """Test default report configuration."""
        config = ReportConfig()
        assert config.out_dir == "./artifacts"
        assert config.format == "html"

    def test_custom_report(self):
        """Test custom report configuration."""
        config = ReportConfig(
            out_dir="/tmp/reports",
            format="md"
        )
        assert config.out_dir == "/tmp/reports"
        assert config.format == "md"


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_default_llm(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.openai_model == "gpt-4o-mini"
        assert config.ollama_model == "llama3.2"

    def test_custom_llm(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            openai_model="gpt-4",
            ollama_model="llama2"
        )
        assert config.openai_model == "gpt-4"
        assert config.ollama_model == "llama2"


class TestConfig:
    """Test main Config model."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.task == "auto"
        assert config.metrics["primary"] == "auto"
        assert config.business_goal == "Plain English goal"

    def test_classification_config(self):
        """Test classification configuration."""
        config = Config(
            task="classification",
            metrics={"primary": "accuracy", "secondary": ["f1", "precision"]},
            business_goal="Classify iris species"
        )
        assert config.task == "classification"
        assert config.metrics["primary"] == "accuracy"
        assert "f1" in config.metrics["secondary"]

    def test_regression_config(self):
        """Test regression configuration."""
        config = Config(
            task="regression",
            metrics={"primary": "mae", "secondary": ["rmse", "r2"]},
            business_goal="Predict diabetes progression"
        )
        assert config.task == "regression"
        assert config.metrics["primary"] == "mae"

    def test_csv_validation_missing_paths(self):
        """Test CSV validation fails without paths."""
        with pytest.raises(ValueError, match="csv_paths required"):
            Config(
                data={"source": "csv", "name": "test", "csv_paths": [], "target": "y"}
            )

    def test_csv_validation_missing_target(self):
        """Test CSV validation fails without target."""
        with pytest.raises(ValueError, match="target required"):
            Config(
                data={"source": "csv", "name": "test", "csv_paths": ["/path/to/data.csv"], "target": ""}
            )

    def test_csv_validation_success(self):
        """Test CSV validation succeeds with all required fields."""
        config = Config(
            data={
                "source": "csv",
                "name": "test",
                "csv_paths": ["/path/to/data.csv"],
                "target": "target_col"
            }
        )
        assert config.data.source == "csv"
        assert config.data.target == "target_col"

    def test_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_content = """
data:
  source: builtin
  name: iris
task: classification
business_goal: "Classify iris species"
metrics:
  primary: accuracy
  secondary:
    - f1
    - precision
budgets:
  time_min: 20
  memory_gb: 2.0
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        
        config = Config.from_yaml(str(yaml_file))
        assert config.data.name == "iris"
        assert config.task == "classification"
        assert config.budgets.time_min == 20

    def test_validate_environment_missing_key(self, monkeypatch):
        """Test environment validation fails without OpenAI key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Config()
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            config.validate_environment()

    def test_validate_environment_success(self, monkeypatch):
        """Test environment validation succeeds with OpenAI key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Config()
        
        # Should not raise
        config.validate_environment()

    def test_nested_config_access(self):
        """Test accessing nested configuration."""
        config = Config()
        assert config.data.source == "builtin"
        assert config.budgets.time_min == 25
        assert config.split.seed == 42
        assert config.cache.mode == "warm"

    def test_config_serialization(self):
        """Test configuration can be serialized."""
        config = Config(
            task="classification",
            business_goal="Test goal"
        )
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["task"] == "classification"
        assert config_dict["business_goal"] == "Test goal"

    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "task": "regression",
            "business_goal": "Predict values",
            "data": {"source": "builtin", "name": "diabetes"},
            "metrics": {"primary": "mae"}
        }
        config = Config(**config_dict)
        
        assert config.task == "regression"
        assert config.data.name == "diabetes"
        assert config.metrics["primary"] == "mae"

