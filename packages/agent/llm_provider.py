"""
LLM Provider module for managing LLM client selection and shadow mode.

Phase 5 Features:
- Provider selection based on APP_ENV and CI detection
- LLM_PROVIDER_USED trace event generation
- Shadow mode for comparing LLM outputs without affecting user responses

Environment Variables:
- APP_ENV: dev|test|staging|prod (determines provider)
- CI: If set, forces fake provider
- SHADOW_MODE_ENABLED: true|false (default false)
- SHADOW_MODE_SAMPLE_RATE: 0.0..1.0 (default 0.0)
- SHADOW_VERTEX_LLM_MODEL: optional override for shadow model
- SHADOW_PROMPT_VERSION: optional prompt version for shadow
"""

import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger(__name__)

# Prompt version for tracking
PROMPT_VERSION = "v1"


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


@dataclass
class LLMProviderInfo:
    """Information about which LLM provider is being used."""
    provider: str  # "vertex" or "fake"
    model: str
    prompt_version: str


@dataclass
class ShadowResult:
    """Result from shadow LLM call."""
    success: bool
    answer_text: str | None
    latency_ms: float
    completion_chars: int
    model: str
    error: str | None = None


def get_app_env() -> str:
    """Get the current application environment."""
    return os.getenv("APP_ENV", "dev").lower()


def is_ci_environment() -> bool:
    """Check if running in CI environment."""
    # Check common CI environment variables
    ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_HOME", "CIRCLECI"]
    return any(os.getenv(var) for var in ci_vars)


def should_use_vertex() -> bool:
    """
    Determine if Vertex AI should be used based on environment.

    Rules:
    - If CI=true or running in CI environment: ALWAYS use fake
    - If APP_ENV=test: ALWAYS use fake
    - If APP_ENV=staging|prod AND Vertex env vars exist: use vertex
    - Otherwise: use fake
    """
    # CI environments always use fake
    if is_ci_environment():
        return False

    env = get_app_env()

    # Test environment always uses fake
    if env == "test":
        return False

    # Staging/prod use Vertex if configured
    if env in ("staging", "prod", "production"):
        # Verify Vertex configuration exists
        gcp_project = os.getenv("GCP_PROJECT", "")
        if gcp_project:
            return True

    return False


def get_shadow_config() -> dict:
    """Get shadow mode configuration from environment."""
    return {
        "enabled": os.getenv("SHADOW_MODE_ENABLED", "false").lower() == "true",
        "sample_rate": float(os.getenv("SHADOW_MODE_SAMPLE_RATE", "0.0")),
        "shadow_model": os.getenv("SHADOW_VERTEX_LLM_MODEL", ""),
        "shadow_prompt_version": os.getenv("SHADOW_PROMPT_VERSION", ""),
    }


def should_run_shadow() -> bool:
    """
    Determine if shadow mode should run for this request.

    Rules:
    - Must be enabled via SHADOW_MODE_ENABLED=true
    - Must pass sample rate check
    - Must be in staging/prod (never in test/dev/CI)
    """
    config = get_shadow_config()

    if not config["enabled"]:
        return False

    env = get_app_env()
    if env not in ("staging", "prod", "production"):
        return False

    if is_ci_environment():
        return False

    # Sample rate check
    if random.random() >= config["sample_rate"]:
        return False

    return True


def get_llm_client_with_info() -> tuple[LLMClient, LLMProviderInfo]:
    """
    Get the appropriate LLM client based on environment.

    Returns:
        Tuple of (client, provider_info)
    """
    from .fake_clients import FakeLLMClient

    if should_use_vertex():
        try:
            from .vertex_clients import VertexLLMClient, get_vertex_config
            config = get_vertex_config()
            client = VertexLLMClient()
            info = LLMProviderInfo(
                provider="vertex",
                model=config["llm_model"],
                prompt_version=PROMPT_VERSION,
            )
            return client, info
        except ImportError:
            logger.warning("Vertex AI not available, falling back to fake client")

    # Fake client
    return FakeLLMClient(), LLMProviderInfo(
        provider="fake",
        model="fake-llm",
        prompt_version=PROMPT_VERSION,
    )


def get_shadow_client() -> tuple[LLMClient | None, str]:
    """
    Get a shadow LLM client if shadow mode should run.

    Returns:
        Tuple of (client or None, model_name)
    """
    if not should_run_shadow():
        return None, ""

    config = get_shadow_config()
    shadow_model = config["shadow_model"]

    try:
        from .vertex_clients import VertexLLMClient, get_vertex_config

        if shadow_model:
            # Use specified shadow model
            client = VertexLLMClient(model=shadow_model)
            return client, shadow_model
        else:
            # Use default model
            vertex_config = get_vertex_config()
            client = VertexLLMClient()
            return client, vertex_config["llm_model"]
    except ImportError:
        logger.warning("Shadow mode: Vertex AI not available")
        return None, ""


def run_shadow_call(
    client: LLMClient,
    prompt: str,
    model: str,
    _timeout_seconds: float = 30.0,  # Reserved for future timeout implementation
) -> ShadowResult:
    """
    Run a shadow LLM call with timeout protection.

    This should never fail the primary request - all errors are captured.
    """
    start_time = time.time()

    try:
        # Run the shadow call
        response = client.generate(prompt)
        latency_ms = (time.time() - start_time) * 1000

        # Truncate response if too long (for trace storage)
        max_chars = 2000
        truncated = response[:max_chars] if len(response) > max_chars else response

        return ShadowResult(
            success=True,
            answer_text=truncated,
            latency_ms=round(latency_ms, 2),
            completion_chars=len(response),
            model=model,
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.warning(f"Shadow LLM call failed: {e}")

        return ShadowResult(
            success=False,
            answer_text=None,
            latency_ms=round(latency_ms, 2),
            completion_chars=0,
            model=model,
            error=str(e),
        )


def compute_shadow_diff(primary_text: str, shadow_text: str | None) -> dict:
    """
    Compute a simple, deterministic diff metric between primary and shadow outputs.

    Uses Jaccard similarity on word sets - cheap and deterministic (no embeddings).
    """
    if shadow_text is None:
        return {
            "similarity_proxy": 0.0,
            "changed_refusal": False,
            "changed_route": False,
            "primary_length": len(primary_text),
            "shadow_length": 0,
        }

    # Tokenize to word sets (simple split, lowercased)
    primary_words = set(primary_text.lower().split())
    shadow_words = set(shadow_text.lower().split())

    # Jaccard similarity
    if not primary_words and not shadow_words:
        similarity = 1.0
    elif not primary_words or not shadow_words:
        similarity = 0.0
    else:
        intersection = len(primary_words & shadow_words)
        union = len(primary_words | shadow_words)
        similarity = round(intersection / union, 4) if union > 0 else 0.0

    # Detect refusal changes (simple heuristics)
    refusal_keywords = ["cannot", "sorry", "unable", "i'm not able", "i can't"]
    primary_has_refusal = any(kw in primary_text.lower() for kw in refusal_keywords)
    shadow_has_refusal = any(kw in shadow_text.lower() for kw in refusal_keywords)

    return {
        "similarity_proxy": similarity,
        "changed_refusal": primary_has_refusal != shadow_has_refusal,
        "changed_route": False,  # Would require route extraction
        "primary_length": len(primary_text),
        "shadow_length": len(shadow_text),
    }


def create_provider_trace_event(info: LLMProviderInfo) -> dict:
    """Create the LLM_PROVIDER_USED trace event."""
    return {
        "event_type": "LLM_PROVIDER_USED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "provider": info.provider,
            "model": info.model,
            "prompt_version": info.prompt_version,
        },
    }


def create_shadow_result_event(result: ShadowResult) -> dict:
    """Create the SHADOW_LLM_RESULT trace event."""
    return {
        "event_type": "SHADOW_LLM_RESULT",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "shadow_model": result.model,
            "latency_ms": result.latency_ms,
            "completion_chars": result.completion_chars,
            "success": result.success,
            "shadow_answer_text": result.answer_text,
            "shadow_error": result.error,
        },
    }


def create_shadow_diff_event(diff: dict) -> dict:
    """Create the SHADOW_DIFF trace event."""
    return {
        "event_type": "SHADOW_DIFF",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": diff,
    }

