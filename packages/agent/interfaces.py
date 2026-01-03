"""
Protocol interfaces for LLM and Embeddings clients.
These allow swapping implementations (Vertex AI, fake, etc.) without changing agent code.
"""

from typing import Protocol


class EmbeddingsClient(Protocol):
    """Protocol for embedding text into vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query into a vector."""
        ...


class LLMClient(Protocol):
    """Protocol for generating text from an LLM."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...

