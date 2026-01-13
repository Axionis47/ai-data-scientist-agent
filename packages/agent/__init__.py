"""
Agent package - LangGraph orchestration with RAG retrieval and EDA tools.

Provides:
- Protocol interfaces for LLM and Embeddings clients
- Fake clients for testing (no network calls)
- Vertex AI clients for production
- LLM provider management with shadow mode
- RAG retrieval with cosine similarity
- LangGraph-based agent graph
- Deterministic EDA tools
"""

from .fake_clients import FakeEmbeddingsClient, FakeLLMClient
from .graph import AgentState, run_agent
from .interfaces import EmbeddingsClient, LLMClient
from .llm_provider import (
    LLMProviderInfo,
    get_llm_client_with_info,
    get_shadow_config,
    should_use_vertex,
)
from .planner import PlaybookSelection, select_playbook
from .retrieval import (
    RetrievalConfig,
    RetrievedChunk,
    embed_and_store_chunks,
    retrieve_for_causal,
    retrieve_top_k,
)
from .tools_eda import (
    EDAToolError,
    correlation,
    dataset_overview,
    groupby_aggregate,
    time_trend,
    univariate_summary,
)
from .versioning import (
    PROMPT_VERSIONS,
    PromptVersion,
    SegmentVersion,
    VersionManifest,
    create_segment_version,
    create_version_manifest,
    get_prompt_version,
)

__all__ = [
    "EmbeddingsClient",
    "LLMClient",
    "FakeEmbeddingsClient",
    "FakeLLMClient",
    # Retrieval
    "RetrievalConfig",
    "RetrievedChunk",
    "embed_and_store_chunks",
    "retrieve_for_causal",
    "retrieve_top_k",
    # Agent
    "AgentState",
    "run_agent",
    # LLM provider
    "LLMProviderInfo",
    "get_llm_client_with_info",
    "get_shadow_config",
    "should_use_vertex",
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
    # Versioning
    "PROMPT_VERSIONS",
    "PromptVersion",
    "SegmentVersion",
    "VersionManifest",
    "create_segment_version",
    "create_version_manifest",
    "get_prompt_version",
]

