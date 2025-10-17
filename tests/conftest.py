"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from botds import Config
from botds.context import DecisionLog, HandoffLedger, RunManifest


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def artifacts_dir(temp_dir: Path) -> Path:
    """Create artifacts directory structure."""
    artifacts = temp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "handoffs").mkdir(exist_ok=True)
    (artifacts / "logs").mkdir(exist_ok=True)
    (artifacts / "reports").mkdir(exist_ok=True)
    (artifacts / "models").mkdir(exist_ok=True)
    return artifacts


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def basic_config() -> Config:
    """Create a basic valid configuration."""
    return Config(
        data={"source": "builtin", "name": "iris", "target": ""},
        task="classification",
        business_goal="Classify iris species accurately",
        metrics={"primary": "accuracy", "secondary": ["f1", "precision"]},
    )


@pytest.fixture
def csv_config(temp_dir: Path) -> Config:
    """Create a configuration for CSV data source."""
    # Create a dummy CSV file
    csv_path = temp_dir / "data.csv"
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    
    return Config(
        data={
            "source": "csv",
            "name": "test_data",
            "csv_paths": [str(csv_path)],
            "target": "target"
        },
        task="classification",
        business_goal="Test classification task",
    )


@pytest.fixture
def regression_config() -> Config:
    """Create a configuration for regression task."""
    return Config(
        data={"source": "builtin", "name": "diabetes", "target": ""},
        task="regression",
        business_goal="Predict diabetes progression",
        metrics={"primary": "mae", "secondary": ["rmse", "r2"]},
    )


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "feature3": ["A", "B", "A", "B", "A"],
        "target": [0, 1, 0, 1, 0]
    })


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """Create a larger DataFrame for testing."""
    import numpy as np
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n)
    })


@pytest.fixture
def classification_dataset() -> tuple[pd.DataFrame, str]:
    """Create a classification dataset."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    return df, "target"


@pytest.fixture
def regression_dataset() -> tuple[pd.DataFrame, str]:
    """Create a regression dataset."""
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    return df, "target"


# ============================================================================
# OpenAI Mocking Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response() -> ChatCompletion:
    """Create a mock OpenAI ChatCompletion response."""
    return ChatCompletion(
        id="chatcmpl-test123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is a test response from OpenAI.",
                ),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


@pytest.fixture
def mock_openai_function_call() -> ChatCompletion:
    """Create a mock OpenAI response with function calling."""
    return ChatCompletion(
        id="chatcmpl-test456",
        object="chat.completion",
        created=1234567890,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        MagicMock(
                            id="call_test123",
                            type="function",
                            function=MagicMock(
                                name="select_features",
                                arguments='{"features": ["feature1", "feature2"]}'
                            )
                        )
                    ]
                ),
                finish_reason="tool_calls",
            )
        ],
        usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    )


@pytest.fixture
def mock_openai_client(mock_openai_response: ChatCompletion):
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    return mock_client


@pytest.fixture
def openai_api_key_env():
    """Set OPENAI_API_KEY environment variable for tests."""
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only"
    yield
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)


# ============================================================================
# Context Fixtures
# ============================================================================

@pytest.fixture
def decision_log(temp_dir: Path) -> DecisionLog:
    """Create a DecisionLog instance."""
    log_path = temp_dir / "decision_log.jsonl"
    return DecisionLog(log_path)


@pytest.fixture
def handoff_ledger(temp_dir: Path) -> HandoffLedger:
    """Create a HandoffLedger instance."""
    ledger_path = temp_dir / "handoff_ledger.jsonl"
    return HandoffLedger(ledger_path)


@pytest.fixture
def run_manifest() -> RunManifest:
    """Create a RunManifest instance."""
    return RunManifest(job_id="test1234")


# ============================================================================
# Tool Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_router(openai_api_key_env, mock_openai_client):
    """Create a mock LLM router."""
    with patch("botds.llm.OpenAI", return_value=mock_openai_client):
        from botds.llm import LLMRouter
        from botds.context import DecisionLog
        
        decision_log = DecisionLog(Path("/tmp/test_decision_log.jsonl"))
        config = {"openai_model": "gpt-4o-mini", "ollama_model": "llama3.2"}
        
        router = LLMRouter(config, decision_log)
        yield router


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Create sample JSON data for testing."""
    return {
        "job_id": "test1234",
        "status": "success",
        "metrics": {
            "accuracy": 0.95,
            "f1": 0.93,
            "precision": 0.94
        },
        "features": ["feature1", "feature2", "feature3"],
        "model": "RandomForest"
    }


@pytest.fixture
def sample_handoff_data() -> Dict[str, Any]:
    """Create sample handoff data."""
    return {
        "stage": "profiling",
        "inputs": {"dataset_hash": "sha256:abc123"},
        "outputs": {
            "n_rows": 150,
            "n_cols": 5,
            "target_col": "target",
            "dtypes": {"feature1": "float64", "target": "int64"}
        },
        "schema": "profile_v1",
        "hash": "sha256:def456",
        "timestamp": "2025-10-17T00:00:00Z"
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require mocked services"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "requires_openai: Tests that require OpenAI API key"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to tests in tests/unit/
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add 'integration' marker to tests in tests/integration/
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

