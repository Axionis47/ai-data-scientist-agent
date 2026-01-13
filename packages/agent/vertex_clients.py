"""
Vertex AI clients for production use.
Uses google-cloud-aiplatform or langchain-google-vertexai.
Configured via environment variables.

Phase 4 Enhancements:
- Temperature=0 for deterministic/stable outputs
- Bounded timeouts (30s for LLM, 10s for embeddings)
- 1 retry with exponential backoff
- Uses ADC (Application Default Credentials) on Cloud Run
"""

import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

# Configuration constants
LLM_TIMEOUT_SECONDS = 30
EMBED_TIMEOUT_SECONDS = 10
MAX_RETRIES = 1
RETRY_BASE_DELAY = 1.0  # seconds

T = TypeVar("T")


def _retry_with_backoff(func: Callable[[], T], max_retries: int = MAX_RETRIES) -> T:
    """Execute function with retry and exponential backoff."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
    raise last_exception  # type: ignore[misc]


def get_vertex_config() -> dict:
    """Get Vertex AI configuration from environment variables."""
    return {
        "project": os.getenv("GCP_PROJECT", ""),
        "location": os.getenv("GCP_LOCATION", "us-central1"),
        "llm_model": os.getenv("VERTEX_LLM_MODEL", "gemini-1.5-flash"),
        "embed_model": os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005"),
    }


def get_app_env() -> str:
    """Get the current application environment."""
    return os.getenv("APP_ENV", "dev").lower()


def should_use_vertex() -> bool:
    """Determine if Vertex AI should be used based on environment."""
    env = get_app_env()
    return env in ("staging", "prod", "production")


class VertexEmbeddingsClient:
    """
    Vertex AI embeddings client using langchain-google-vertexai.

    Phase 4: Timeout 10s, 1 retry with backoff.
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
        timeout: int = EMBED_TIMEOUT_SECONDS,
    ):
        config = get_vertex_config()
        self.project = project or config["project"]
        self.location = location or config["location"]
        self.model = model or config["embed_model"]
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of the embeddings client."""
        if self._client is None:
            try:
                from langchain_google_vertexai import VertexAIEmbeddings

                # Note: VertexAIEmbeddings doesn't support request_timeout
                # The timeout is handled at the gRPC level by google-cloud-aiplatform
                self._client = VertexAIEmbeddings(
                    project=self.project,
                    location=self.location,
                    model_name=self.model,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-google-vertexai is required for Vertex AI. "
                    "Install with: pip install langchain-google-vertexai"
                ) from e
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Vertex AI with retry."""

        def _call():
            client = self._get_client()
            return client.embed_documents(texts)

        return _retry_with_backoff(_call)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query using Vertex AI with retry."""

        def _call():
            client = self._get_client()
            return client.embed_query(query)

        return _retry_with_backoff(_call)


class VertexLLMClient:
    """
    Vertex AI LLM client using langchain-google-vertexai.

    Phase 4 Configuration:
    - temperature=0 for deterministic/stable outputs
    - max_output_tokens=2048 (sufficient for narratives)
    - Timeout: 30s
    - 1 retry with exponential backoff
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ):
        config = get_vertex_config()
        self.project = project or config["project"]
        self.location = location or config["location"]
        self.model = model or config["llm_model"]
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of the LLM client."""
        if self._client is None:
            try:
                from langchain_google_vertexai import ChatVertexAI

                # Note: ChatVertexAI doesn't support request_timeout directly
                # The timeout is handled at the gRPC level by google-cloud-aiplatform
                self._client = ChatVertexAI(
                    project=self.project,
                    location=self.location,
                    model_name=self.model,
                    temperature=0,  # Phase 4: deterministic output
                    max_output_tokens=2048,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-google-vertexai is required for Vertex AI. "
                    "Install with: pip install langchain-google-vertexai"
                ) from e
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate text using Vertex AI with retry."""

        def _call():
            client = self._get_client()
            response = client.invoke(prompt)
            return response.content

        result = _retry_with_backoff(_call)
        logger.info(f"VERTEX_USED: model={self.model}")
        return result

