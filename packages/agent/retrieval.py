"""
RAG retrieval module.
Handles embedding storage and cosine similarity retrieval.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .interfaces import EmbeddingsClient


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


def retrieve_top_k(
    doc_id: str,
    query: str,
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
    k: int = 4,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k most similar chunks for a query.

    Args:
        doc_id: Document ID
        query: Query string
        embeddings_client: Client to embed the query
        storage_dir: Base storage directory
        k: Number of chunks to retrieve

    Returns:
        List of RetrievedChunk objects sorted by similarity (highest first)
    """
    # Load stored embeddings
    embedded_chunks = load_embeddings(doc_id, storage_dir)

    # Embed the query
    query_embedding = embeddings_client.embed_query(query)

    # Compute similarities
    scored_chunks = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((chunk, score))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top-k
    results = []
    for chunk, score in scored_chunks[:k]:
        results.append(RetrievedChunk(
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            score=score,
            section=chunk.get("section"),
        ))

    return results

