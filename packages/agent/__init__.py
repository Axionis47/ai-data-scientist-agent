"""
Agent package - LangGraph orchestration with RAG retrieval.

Provides:
- Protocol interfaces for LLM and Embeddings clients
- Fake clients for testing (no network calls)
- Vertex AI clients for production
- RAG retrieval with cosine similarity
- LangGraph-based agent graph
"""

from .fake_clients import FakeEmbeddingsClient, FakeLLMClient
from .graph import AgentState, run_agent
from .interfaces import EmbeddingsClient, LLMClient
from .retrieval import RetrievedChunk, embed_and_store_chunks, retrieve_top_k

__all__ = [
    "EmbeddingsClient",
    "LLMClient",
    "FakeEmbeddingsClient",
    "FakeLLMClient",
    "RetrievedChunk",
    "embed_and_store_chunks",
    "retrieve_top_k",
    "AgentState",
    "run_agent",
]

