"""
Versioning system for segments, prompts, and API components.

Provides semantic versioning for:
- Document segments/chunks (content hash + chunking parameters)
- Prompt templates (version string + hash)
- Embedding models (model name + version)

This enables:
- Reproducibility of results
- A/B testing of prompt versions
- Tracking which version produced which output
- Cache invalidation when parameters change
"""

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class SegmentVersion:
    """Version information for a document segment/chunk."""
    segment_id: str  # Unique ID for this segment
    doc_id: str  # Parent document ID
    chunk_index: int
    content_hash: str  # SHA-256 of chunk text
    version: str  # Semantic version (e.g., "1.0.0")
    created_at: str  # ISO timestamp
    chunking_params: dict = field(default_factory=dict)  # chunk_size, overlap, etc.
    embedding_model: str = ""  # e.g., "text-embedding-005"


@dataclass
class PromptVersion:
    """Version information for a prompt template."""
    prompt_id: str  # e.g., "system_analysis", "causal_gate"
    version: str  # Semantic version (e.g., "v1.0.0")
    content_hash: str  # SHA-256 of prompt template
    description: str = ""
    created_at: str = ""


# Current prompt versions - increment these when prompts change
PROMPT_VERSIONS = {
    "system_analysis": PromptVersion(
        prompt_id="system_analysis",
        version="v1.0.0",
        content_hash="",  # Computed at runtime
        description="Base system prompt for analysis questions",
    ),
    "causal_gate": PromptVersion(
        prompt_id="causal_gate",
        version="v1.0.0",
        content_hash="",
        description="Prompt for causal readiness assessment",
    ),
    "causal_estimation": PromptVersion(
        prompt_id="causal_estimation",
        version="v1.0.0",
        content_hash="",
        description="Prompt for causal effect estimation",
    ),
}


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_segment_version(
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    embedding_model: str = "text-embedding-005",
) -> SegmentVersion:
    """Create a version record for a document segment."""
    content_hash = compute_content_hash(chunk_text)
    segment_id = f"{doc_id}:{chunk_index}:{content_hash[:8]}"

    return SegmentVersion(
        segment_id=segment_id,
        doc_id=doc_id,
        chunk_index=chunk_index,
        content_hash=content_hash,
        version="1.0.0",  # Initial version
        created_at=datetime.now(UTC).isoformat(),
        chunking_params={
            "chunk_size": chunk_size,
            "overlap": overlap,
            "algorithm": "sliding_window",
        },
        embedding_model=embedding_model,
    )


def get_prompt_version(prompt_id: str) -> PromptVersion:
    """Get the current version of a prompt template."""
    if prompt_id in PROMPT_VERSIONS:
        return PROMPT_VERSIONS[prompt_id]
    return PromptVersion(
        prompt_id=prompt_id,
        version="v0.0.0",
        content_hash="unknown",
        description="Unknown prompt",
    )


def update_prompt_hash(prompt_id: str, template_content: str) -> PromptVersion:
    """Update the content hash for a prompt (called when prompt is used)."""
    if prompt_id in PROMPT_VERSIONS:
        pv = PROMPT_VERSIONS[prompt_id]
        pv.content_hash = compute_content_hash(template_content)
        pv.created_at = datetime.now(UTC).isoformat()
        return pv
    return get_prompt_version(prompt_id)


@dataclass
class VersionManifest:
    """Complete version manifest for a request/response."""
    api_version: str = "0.1.0"
    prompt_versions: dict = field(default_factory=dict)
    segment_versions: list = field(default_factory=list)
    embedding_model: str = ""
    llm_model: str = ""
    timestamp: str = ""


def create_version_manifest(
    prompt_ids: list[str],
    segment_versions: list[SegmentVersion] | None = None,
    embedding_model: str = "",
    llm_model: str = "",
) -> VersionManifest:
    """Create a version manifest capturing all component versions."""
    prompt_versions = {
        pid: get_prompt_version(pid).__dict__
        for pid in prompt_ids
    }

    segment_dicts = [s.__dict__ for s in (segment_versions or [])]

    return VersionManifest(
        api_version="0.1.0",
        prompt_versions=prompt_versions,
        segment_versions=segment_dicts,
        embedding_model=embedding_model,
        llm_model=llm_model,
        timestamp=datetime.now(UTC).isoformat(),
    )

