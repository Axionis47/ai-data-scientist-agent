"""
LangGraph-based agent graph for RAG question answering.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .interfaces import EmbeddingsClient, LLMClient
from .retrieval import retrieve_top_k


class AgentState(TypedDict, total=False):
    """State passed through the agent graph (TypedDict for LangGraph)."""
    # Input
    question: str
    doc_id: str
    dataset_id: str | None
    session_id: str | None

    # Computed
    route: str
    route_confidence: float
    route_reasons: list[str]
    retrieved_chunks: list[dict]  # Serialized RetrievedChunk
    prompt: str
    llm_response: str
    trace_id: str
    trace_events: list[dict]

    # Artifacts
    artifacts: list[dict]


def add_trace_event(state: AgentState, event_type: str, payload: dict | None = None) -> None:
    """Add a trace event to the state."""
    if "trace_events" not in state:
        state["trace_events"] = []
    state["trace_events"].append({
        "event_type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": payload or {},
    })


# Router patterns for deterministic classification
CAUSAL_PATTERNS = [
    r"\beffect\b", r"\bimpact\b", r"\bcausal\b", r"\bate\b",
    r"\bcauses?\b", r"\binfluence\b", r"\baffect\b",
]
REPORTING_PATTERNS = [
    r"\breport\b", r"\bexport\b", r"\bsummary\s+doc\b", r"\bgenerate\s+report\b",
]


def route_node(state: AgentState) -> dict:
    """Deterministic router based on keyword patterns."""
    question_lower = state["question"].lower()
    artifacts = state.get("artifacts", [])
    trace_events = state.get("trace_events", [])

    # Check for causal patterns
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, question_lower):
            # In Phase 1, causal queries return NEEDS_CLARIFICATION
            route = "NEEDS_CLARIFICATION"
            reasons = [f"Matched causal pattern: {pattern}", "Causal analysis requires clarification (Phase 1 placeholder)"]
            artifacts = artifacts + [{
                "type": "checklist",
                "items": [
                    "Specify the treatment variable",
                    "Specify the outcome variable",
                    "Identify potential confounders",
                    "Confirm causal assumptions",
                ],
            }]
            trace_events = trace_events + [{
                "event_type": "ROUTED",
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": {"route": route, "reason": "causal_pattern"},
            }]
            return {"route": route, "route_confidence": 0.9, "route_reasons": reasons, "artifacts": artifacts, "trace_events": trace_events}

    # Check for reporting patterns
    for pattern in REPORTING_PATTERNS:
        if re.search(pattern, question_lower):
            route = "REPORTING"
            reasons = [f"Matched reporting pattern: {pattern}"]
            artifacts = artifacts + [{
                "type": "text",
                "content": "Reporting/export functionality coming in a future phase.",
            }]
            trace_events = trace_events + [{
                "event_type": "ROUTED",
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": {"route": route, "reason": "reporting_pattern"},
            }]
            return {"route": route, "route_confidence": 0.9, "route_reasons": reasons, "artifacts": artifacts, "trace_events": trace_events}

    # Default to ANALYSIS
    trace_events = trace_events + [{
        "event_type": "ROUTED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {"route": "ANALYSIS", "reason": "default"},
    }]
    return {
        "route": "ANALYSIS",
        "route_confidence": 1.0,
        "route_reasons": ["Default route for analytical questions"],
        "trace_events": trace_events,
    }


def retrieve_context_node(
    state: AgentState,
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
    k: int = 4,
) -> dict:
    """Retrieve relevant chunks from the context document."""
    trace_events = state.get("trace_events", [])

    try:
        chunks = retrieve_top_k(
            doc_id=state["doc_id"],
            query=state["question"],
            embeddings_client=embeddings_client,
            storage_dir=storage_dir,
            k=k,
        )
        # Serialize chunks to dict
        retrieved = [{"chunk_index": c.chunk_index, "text": c.text, "score": c.score, "section": c.section} for c in chunks]
        trace_events = trace_events + [{
            "event_type": "RETRIEVED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "num_chunks": len(chunks),
                "chunk_indices": [c.chunk_index for c in chunks],
                "scores": [c.score for c in chunks],
            },
        }]
        return {"retrieved_chunks": retrieved, "trace_events": trace_events}
    except FileNotFoundError:
        trace_events = trace_events + [{
            "event_type": "RETRIEVED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": "embeddings_not_found"},
        }]
        return {"retrieved_chunks": [], "trace_events": trace_events}


def compose_prompt_node(state: AgentState) -> dict:
    """Compose the prompt for the LLM with retrieved context."""
    system_instruction = (
        "You are a helpful data analysis assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the information is not present in the context, say you don't know. "
        "Be concise and accurate."
    )

    # Format retrieved chunks
    context_parts = []
    for chunk in state.get("retrieved_chunks", []):
        context_parts.append(f"[Chunk {chunk['chunk_index']}]: {chunk['text']}")

    context_text = "\n\n".join(context_parts) if context_parts else "(No context available)"

    prompt = f"""{system_instruction}

CONTEXT:
{context_text}

User question: {state["question"]}

Answer:"""

    return {"prompt": prompt}


def call_llm_node(state: AgentState, llm_client: LLMClient) -> dict:
    """Call the LLM with the composed prompt."""
    trace_events = state.get("trace_events", [])
    prompt = state.get("prompt", "")

    try:
        response = llm_client.generate(prompt)
        trace_events = trace_events + [{
            "event_type": "LLM_CALLED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "prompt_length": len(prompt),
                "response_length": len(response),
            },
        }]
        return {"llm_response": response, "trace_events": trace_events}
    except Exception as e:
        trace_events = trace_events + [{
            "event_type": "LLM_CALLED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": str(e)},
        }]
        return {"llm_response": f"Error generating response: {e}", "trace_events": trace_events}


def build_response_node(state: AgentState) -> dict:
    """Build the final response with artifacts."""
    artifacts = state.get("artifacts", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Add retrieved chunks artifact for transparency
    if retrieved_chunks:
        chunk_summaries = []
        for chunk in retrieved_chunks:
            text = chunk.get("text", "")
            excerpt = text[:150] + "..." if len(text) > 150 else text
            chunk_summaries.append(f"Chunk {chunk.get('chunk_index')} (score: {chunk.get('score', 0):.3f}): {excerpt}")

        artifacts = artifacts + [{
            "type": "text",
            "content": "Retrieved context chunks:\n" + "\n".join(chunk_summaries),
        }]

    return {"artifacts": artifacts}


def create_agent_graph(
    embeddings_client: EmbeddingsClient,
    llm_client: LLMClient,
    storage_dir: Path,
) -> StateGraph:
    """
    Create the LangGraph agent graph.

    Args:
        embeddings_client: Client for embeddings
        llm_client: Client for LLM generation
        storage_dir: Path to document storage

    Returns:
        Compiled StateGraph
    """
    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("retrieve_context", lambda s: retrieve_context_node(s, embeddings_client, storage_dir))
    graph.add_node("compose_prompt", compose_prompt_node)
    graph.add_node("call_llm", lambda s: call_llm_node(s, llm_client))
    graph.add_node("build_response", build_response_node)

    # Define edges
    graph.set_entry_point("route")

    # Conditional routing based on route type
    def should_continue_to_rag(state: AgentState) -> str:
        if state.get("route") in ["NEEDS_CLARIFICATION", "REPORTING"]:
            return "build_response"
        return "retrieve_context"

    graph.add_conditional_edges(
        "route",
        should_continue_to_rag,
        {
            "retrieve_context": "retrieve_context",
            "build_response": "build_response",
        }
    )

    graph.add_edge("retrieve_context", "compose_prompt")
    graph.add_edge("compose_prompt", "call_llm")
    graph.add_edge("call_llm", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()


def run_agent(
    question: str,
    doc_id: str,
    embeddings_client: EmbeddingsClient,
    llm_client: LLMClient,
    storage_dir: Path,
    dataset_id: str | None = None,
    session_id: str | None = None,
) -> AgentState:
    """
    Run the agent graph and return the final state.

    Args:
        question: User's question
        doc_id: Document ID for context
        embeddings_client: Embeddings client
        llm_client: LLM client
        storage_dir: Storage directory path
        dataset_id: Optional dataset ID
        session_id: Optional session ID

    Returns:
        Final AgentState (dict) with response
    """
    import uuid as uuid_module

    graph = create_agent_graph(embeddings_client, llm_client, storage_dir)

    initial_state: AgentState = {
        "question": question,
        "doc_id": doc_id,
        "dataset_id": dataset_id,
        "session_id": session_id,
        "trace_id": str(uuid_module.uuid4()),
        "trace_events": [],
        "artifacts": [],
        "route": "",
        "route_confidence": 0.0,
        "route_reasons": [],
        "retrieved_chunks": [],
        "prompt": "",
        "llm_response": "",
    }

    final_state = graph.invoke(initial_state)
    return final_state

