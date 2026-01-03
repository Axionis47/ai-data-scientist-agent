"""
Fake/mock clients for testing without network calls.
These provide deterministic behavior for CI/local testing.
"""

import hashlib


class FakeEmbeddingsClient:
    """
    Fake embeddings client that produces deterministic vectors via hashing.
    Each text is hashed and converted to a fixed-length float vector.
    """

    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic vector using SHA256 hash."""
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Extend hash to fill vector dimension
        extended = hash_bytes * (self.vector_dim // len(hash_bytes) + 1)
        # Convert to floats in range [-1, 1]
        vector = []
        for i in range(self.vector_dim):
            # Normalize byte value to [-1, 1]
            val = (extended[i] / 127.5) - 1.0
            vector.append(val)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into deterministic vectors."""
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query into a deterministic vector."""
        return self._text_to_vector(query)


class FakeLLMClient:
    """
    Fake LLM client that returns deterministic responses.
    References provided context in its response.
    """

    def generate(self, prompt: str) -> str:
        """Generate a deterministic response that references the context."""
        # Extract context from prompt if present
        if "Context:" in prompt or "CONTEXT:" in prompt:
            # Find question in prompt
            question_marker = None
            for marker in ["Question:", "QUESTION:", "User question:"]:
                if marker in prompt:
                    question_marker = marker
                    break

            if question_marker:
                q_start = prompt.find(question_marker)
                question_excerpt = prompt[q_start:q_start + 100].replace("\n", " ")[:80]
                return (
                    f"Based on the provided context, I found relevant information. "
                    f"The context contains details about the dataset including its overview, "
                    f"data dictionary, and known caveats. "
                    f"[FAKE_LLM_RESPONSE for: {question_excerpt}...]"
                )

        return (
            "I cannot find specific information to answer this question in the provided context. "
            "[FAKE_LLM_RESPONSE - no context provided]"
        )

