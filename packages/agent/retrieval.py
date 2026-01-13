"""
RAG retrieval module.
Handles embedding storage and cosine similarity retrieval.

Improvements in this version:
- Minimum score threshold to filter low-quality matches
- Dynamic k based on question complexity
- Section-aware retrieval with section boosting
- Retrieval configuration dataclass for flexibility
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .interfaces import EmbeddingsClient

# Default thresholds
# Note: 0.1 works well with fake embeddings in tests
# In production with Vertex AI, similarity scores are typically higher
DEFAULT_MIN_SCORE = 0.1  # Chunks below this score are filtered out
DEFAULT_K = 4  # Default number of chunks to retrieve
MAX_K = 10  # Maximum chunks even for complex questions


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""
    k: int = DEFAULT_K
    min_score: float = DEFAULT_MIN_SCORE
    boost_sections: list[str] = field(default_factory=list)  # Sections to boost for this query
    section_boost_factor: float = 1.2  # Multiply score by this for matching sections


@dataclass
class RetrievedChunk:
    """A chunk retrieved via similarity search."""
    chunk_index: int
    text: str
    score: float
    section: str | None = None


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def embed_and_store_chunks(
    doc_id: str,
    chunks: list[dict],
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
) -> Path:
    """
    Embed chunks and store embeddings to disk.

    Args:
        doc_id: Document ID
        chunks: List of chunk dicts with chunk_index and text
        embeddings_client: Client to generate embeddings
        storage_dir: Base storage directory

    Returns:
        Path to the embeddings.json file
    """
    doc_dir = storage_dir / doc_id

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = embeddings_client.embed_texts(texts)

    # Build storage format
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings, strict=False):
        embedded_chunks.append({
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "section": chunk.get("section"),
            "embedding": embedding,
        })

    # Save to disk
    embeddings_path = doc_dir / "embeddings.json"
    with open(embeddings_path, "w") as f:
        json.dump(embedded_chunks, f)

    return embeddings_path


def load_embeddings(doc_id: str, storage_dir: Path) -> list[dict]:
    """Load embeddings from disk for a document."""
    embeddings_path = storage_dir / doc_id / "embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found for doc_id: {doc_id}")

    with open(embeddings_path) as f:
        return json.load(f)


def estimate_query_complexity(query: str) -> int:
    """
    Estimate query complexity to determine how many chunks to retrieve.

    Returns a k value between DEFAULT_K and MAX_K.

    Heuristics:
    - Longer queries typically need more context
    - Questions with multiple parts need more chunks
    - Causal/analytical questions need more context
    """
    query_lower = query.lower()

    # Base k
    k = DEFAULT_K

    # Longer queries get more chunks
    word_count = len(query.split())
    if word_count > 20:
        k += 2
    elif word_count > 10:
        k += 1

    # Questions with conjunctions suggest multiple parts
    if " and " in query_lower or " or " in query_lower:
        k += 1

    # Causal questions often need more context (confounders, assumptions)
    causal_keywords = ["effect", "impact", "causal", "cause", "confound", "bias"]
    if any(kw in query_lower for kw in causal_keywords):
        k += 2

    # Questions about definitions or dictionary need Data Dictionary section
    if "what is" in query_lower or "define" in query_lower or "meaning" in query_lower:
        k += 1

    return min(k, MAX_K)


def detect_relevant_sections(query: str) -> list[str]:
    """
    Detect which sections are most relevant to a query.

    Used to boost scores for chunks from relevant sections.
    """
    query_lower = query.lower()
    sections = []

    # Causal questions → Known Caveats, Causal Assumptions
    if any(kw in query_lower for kw in ["effect", "impact", "causal", "cause"]):
        sections.extend(["Known Caveats", "Causal Assumptions", "Limitations"])

    # Definition questions → Data Dictionary
    if any(kw in query_lower for kw in ["what is", "define", "meaning", "variable"]):
        sections.extend(["Data Dictionary", "Variable Definitions"])

    # Overview questions → Dataset Overview
    if any(kw in query_lower for kw in ["overview", "about", "describe", "summary"]):
        sections.extend(["Dataset Overview", "Target Use / Primary Questions"])

    # Data collection questions
    if any(kw in query_lower for kw in ["collect", "source", "where", "how"]):
        sections.extend(["Data Collection", "Dataset Overview"])

    return sections


def retrieve_top_k(
    doc_id: str,
    query: str,
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
    k: int | None = None,
    config: RetrievalConfig | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k most similar chunks for a query.

    Args:
        doc_id: Document ID
        query: Query string
        embeddings_client: Client to embed the query
        storage_dir: Base storage directory
        k: Number of chunks to retrieve (if None, estimates from query complexity)
        config: Optional RetrievalConfig for fine-grained control

    Returns:
        List of RetrievedChunk objects sorted by similarity (highest first)
        Filtered by minimum score threshold.
    """
    # Build config
    if config is None:
        config = RetrievalConfig()
        config.boost_sections = detect_relevant_sections(query)

    # Determine k - use query complexity if not specified, cap at MAX_K otherwise
    k = estimate_query_complexity(query) if k is None else min(k, MAX_K)

    # Load stored embeddings
    embedded_chunks = load_embeddings(doc_id, storage_dir)

    # Embed the query
    query_embedding = embeddings_client.embed_query(query)

    # Compute similarities with section boosting
    scored_chunks = []
    for chunk in embedded_chunks:
        base_score = cosine_similarity(query_embedding, chunk["embedding"])

        # Apply section boost if chunk is in a relevant section
        chunk_section = chunk.get("section")
        if chunk_section and chunk_section in config.boost_sections:
            score = base_score * config.section_boost_factor
        else:
            score = base_score

        # Store both raw and boosted score for transparency
        scored_chunks.append((chunk, score, base_score))

    # Sort by boosted score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top-k chunks above minimum score threshold
    results = []
    for chunk, score, base_score in scored_chunks:
        # Skip low-scoring chunks
        if base_score < config.min_score:
            continue

        if len(results) >= k:
            break

        results.append(RetrievedChunk(
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            score=round(score, 4),  # Round for determinism
            section=chunk.get("section"),
        ))

    return results


def retrieve_for_causal(
    doc_id: str,
    query: str,
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
) -> list[RetrievedChunk]:
    """
    Specialized retrieval for causal questions.

    Prioritizes chunks from sections about assumptions, caveats, and confounders.
    Uses higher k and lower min_score since causal context is critical.
    """
    config = RetrievalConfig(
        k=8,  # More chunks for causal
        min_score=0.2,  # Lower threshold - causal context is important
        boost_sections=[
            "Known Caveats",
            "Causal Assumptions",
            "Limitations",
            "Data Dictionary",  # Often describes confounders
        ],
        section_boost_factor=1.3,  # Stronger boost for relevant sections
    )

    return retrieve_top_k(
        doc_id=doc_id,
        query=query,
        embeddings_client=embeddings_client,
        storage_dir=storage_dir,
        config=config,
    )

