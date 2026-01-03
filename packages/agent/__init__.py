"""
Agent package - LangGraph orchestration with RAG retrieval and EDA tools.

Provides:
- Protocol interfaces for LLM and Embeddings clients
- Fake clients for testing (no network calls)
- Vertex AI clients for production
- RAG retrieval with cosine similarity
- LangGraph-based agent graph
- Deterministic EDA tools
"""

from .fake_clients import FakeEmbeddingsClient, FakeLLMClient
from .graph import AgentState, run_agent
from .interfaces import EmbeddingsClient, LLMClient
from .retrieval import RetrievedChunk, embed_and_store_chunks, retrieve_top_k
from .planner import PlaybookSelection, select_playbook
from .tools_eda import (
    EDAToolError,
    correlation,
    dataset_overview,
    groupby_aggregate,
    time_trend,
    univariate_summary,
)

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
    # Planner
    "PlaybookSelection",
    "select_playbook",
    # EDA tools
    "EDAToolError",
    "dataset_overview",
    "univariate_summary",
    "groupby_aggregate",
    "time_trend",
    "correlation",
]

