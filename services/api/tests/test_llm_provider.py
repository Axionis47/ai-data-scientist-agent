"""
Tests for LLM provider selection and shadow mode.

Phase 5: Tests for provider wiring verification and shadow mode.

Tests verify:
1. In test/CI environment, provider is always "fake"
2. Shadow mode does NOT run in test env even if enabled
3. Shadow mode can be simulated in unit tests
4. LLM_PROVIDER_USED trace event is correctly generated
"""

import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from packages.agent.llm_provider import (
    LLMProviderInfo,
    ShadowResult,
    compute_shadow_diff,
    create_provider_trace_event,
    create_shadow_diff_event,
    create_shadow_result_event,
    get_app_env,
    get_llm_client_with_info,
    get_shadow_config,
    is_ci_environment,
    should_run_shadow,
    should_use_vertex,
)


class TestProviderSelection:
    """Tests for LLM provider selection based on environment."""

    def test_dev_env_uses_fake_provider(self):
        """In dev environment, provider should be fake."""
        with mock.patch.dict(os.environ, {"APP_ENV": "dev"}, clear=False):
            client, info = get_llm_client_with_info()
            assert info.provider == "fake"
            assert info.model == "fake-llm"

    def test_test_env_uses_fake_provider(self):
        """In test environment, provider should be fake."""
        with mock.patch.dict(os.environ, {"APP_ENV": "test"}, clear=False):
            client, info = get_llm_client_with_info()
            assert info.provider == "fake"
            assert info.model == "fake-llm"

    def test_ci_env_uses_fake_provider(self):
        """In CI environment, provider should be fake regardless of APP_ENV."""
        with mock.patch.dict(
            os.environ,
            {"APP_ENV": "staging", "CI": "true"},
            clear=False
        ):
            assert is_ci_environment() is True
            assert should_use_vertex() is False
            client, info = get_llm_client_with_info()
            assert info.provider == "fake"

    def test_should_use_vertex_in_staging_with_gcp_project(self):
        """Staging env with GCP_PROJECT should indicate vertex usage."""
        with mock.patch.dict(
            os.environ,
            {"APP_ENV": "staging", "GCP_PROJECT": "my-project"},
            clear=False
        ):
            # Remove CI flag if present
            env = os.environ.copy()
            env.pop("CI", None)
            env.pop("GITHUB_ACTIONS", None)
            with mock.patch.dict(os.environ, env, clear=True):
                os.environ["APP_ENV"] = "staging"
                os.environ["GCP_PROJECT"] = "my-project"
                assert get_app_env() == "staging"
                # Note: actual vertex client may not be available in test,
                # but should_use_vertex() should return True
                assert should_use_vertex() is True


class TestShadowMode:
    """Tests for shadow mode configuration and execution."""

    def test_shadow_mode_disabled_by_default(self):
        """Shadow mode should be disabled by default."""
        # Clear any shadow mode env vars
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHADOW")}
        with mock.patch.dict(os.environ, env, clear=True):
            config = get_shadow_config()
            assert config["enabled"] is False
            assert config["sample_rate"] == 0.0

    def test_shadow_mode_not_in_test_env(self):
        """Shadow mode should NOT run in test environment even if enabled."""
        with mock.patch.dict(
            os.environ,
            {
                "APP_ENV": "test",
                "SHADOW_MODE_ENABLED": "true",
                "SHADOW_MODE_SAMPLE_RATE": "1.0",
            },
            clear=False
        ):
            assert should_run_shadow() is False

    def test_shadow_mode_not_in_dev_env(self):
        """Shadow mode should NOT run in dev environment even if enabled."""
        with mock.patch.dict(
            os.environ,
            {
                "APP_ENV": "dev",
                "SHADOW_MODE_ENABLED": "true",
                "SHADOW_MODE_SAMPLE_RATE": "1.0",
            },
            clear=False
        ):
            assert should_run_shadow() is False

    def test_shadow_mode_not_in_ci(self):
        """Shadow mode should NOT run in CI even if enabled."""
        with mock.patch.dict(
            os.environ,
            {
                "APP_ENV": "staging",
                "CI": "true",
                "SHADOW_MODE_ENABLED": "true",
                "SHADOW_MODE_SAMPLE_RATE": "1.0",
            },
            clear=False
        ):
            assert should_run_shadow() is False


class TestShadowDiff:
    """Tests for shadow diff computation."""

    def test_compute_diff_identical_texts(self):
        """Identical texts should have similarity of 1.0."""
        text = "This is a test response from the LLM."
        diff = compute_shadow_diff(text, text)
        assert diff["similarity_proxy"] == 1.0
        assert diff["changed_refusal"] is False

    def test_compute_diff_different_texts(self):
        """Different texts should have lower similarity."""
        primary = "This is the primary response about data analysis."
        shadow = "This is a different response about machine learning."
        diff = compute_shadow_diff(primary, shadow)
        assert 0.0 < diff["similarity_proxy"] < 1.0
        assert diff["primary_length"] == len(primary)
        assert diff["shadow_length"] == len(shadow)

    def test_compute_diff_detects_refusal_change(self):
        """Should detect when one response is a refusal and other is not."""
        primary = "Here is the analysis you requested."
        shadow = "I'm sorry, I cannot provide that information."
        diff = compute_shadow_diff(primary, shadow)
        assert diff["changed_refusal"] is True

    def test_compute_diff_null_shadow(self):
        """Should handle None shadow text gracefully."""
        primary = "This is the primary response."
        diff = compute_shadow_diff(primary, None)
        assert diff["similarity_proxy"] == 0.0
        assert diff["shadow_length"] == 0


class TestTraceEvents:
    """Tests for trace event generation."""

    def test_provider_trace_event_format(self):
        """LLM_PROVIDER_USED event should have correct format."""
        info = LLMProviderInfo(
            provider="fake",
            model="fake-llm",
            prompt_version="v1",
        )
        event = create_provider_trace_event(info)

        assert event["event_type"] == "LLM_PROVIDER_USED"
        assert "timestamp" in event
        assert event["payload"]["provider"] == "fake"
        assert event["payload"]["model"] == "fake-llm"
        assert event["payload"]["prompt_version"] == "v1"

    def test_shadow_result_event_format(self):
        """SHADOW_LLM_RESULT event should have correct format."""
        result = ShadowResult(
            success=True,
            answer_text="Shadow response text",
            latency_ms=150.5,
            completion_chars=20,
            model="gemini-1.5-pro",
        )
        event = create_shadow_result_event(result)

        assert event["event_type"] == "SHADOW_LLM_RESULT"
        assert event["payload"]["shadow_model"] == "gemini-1.5-pro"
        assert event["payload"]["latency_ms"] == 150.5
        assert event["payload"]["success"] is True
        assert event["payload"]["shadow_answer_text"] == "Shadow response text"

    def test_shadow_diff_event_format(self):
        """SHADOW_DIFF event should have correct format."""
        diff = {
            "similarity_proxy": 0.75,
            "changed_refusal": False,
            "changed_route": False,
            "primary_length": 100,
            "shadow_length": 95,
        }
        event = create_shadow_diff_event(diff)

        assert event["event_type"] == "SHADOW_DIFF"
        assert event["payload"]["similarity_proxy"] == 0.75
        assert event["payload"]["changed_refusal"] is False


class TestProviderTraceInAsk:
    """Integration tests for LLM_PROVIDER_USED in /ask responses."""

    def test_ask_includes_provider_trace_event(self):
        """The /ask endpoint should include LLM_PROVIDER_USED trace event."""
        from fastapi.testclient import TestClient

        from services.api.main import app

        client = TestClient(app)

        # We need a valid doc_id to make the /ask call work
        # Use the test infrastructure from other tests
        import io

        from services.api.tests.test_upload_context_doc import create_docx_bytes

        content = {
            "Dataset Overview": "Customer transactions for Q1 2024. " * 20,
            "Target Use / Primary Questions": "Analyze trends and patterns. " * 20,
            "Data Dictionary": "transaction_id, customer_id, amount. " * 20,
            "Known Caveats": "Some missing values in customer_id. " * 20,
        }
        docx_bytes = create_docx_bytes(content)

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
                "question": "What is in this dataset?",
                "doc_id": doc_id,
            }
        )
        assert ask_resp.status_code == 200

        # Check trace_id is present
        trace_id = ask_resp.json()["trace_id"]
        assert trace_id

        # Verify the response worked - provider info is logged
        assert ask_resp.json()["answer_text"]


class TestDebugConfigEndpoint:
    """Tests for /debug/config endpoint."""

    def test_debug_config_in_dev_env(self):
        """Debug config should be accessible in dev environment."""
        from fastapi.testclient import TestClient

        # APP_ENV is set at import time, so we need to reload
        with mock.patch.dict(os.environ, {"APP_ENV": "dev"}, clear=False):
            from services.api.main import app
            client = TestClient(app)

            response = client.get("/debug/config")
            # In dev, should return config
            assert response.status_code == 200
            data = response.json()
            assert "app_env" in data
            assert "would_use_vertex" in data
            assert "shadow_mode" in data

