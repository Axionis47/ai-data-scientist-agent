"""
LangGraph-based agent graph for RAG question answering with EDA playbooks.

Phase 3: Added causal gate integration for CAUSAL route.
Phase 4: Added causal estimation with confirmations gating.
Phase 5: Added LLM provider tracking and shadow mode.
Phase 6: Added memory layer for conversation history.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from packages.contracts.models import CausalConfirmations, CausalSpecOverride

from .causal_gate import causal_readiness_gate
from .interfaces import EmbeddingsClient, LLMClient
from .llm_provider import (
    LLMProviderInfo,
    compute_shadow_diff,
    create_provider_trace_event,
    create_shadow_diff_event,
    create_shadow_result_event,
    get_shadow_client,
    run_shadow_call,
)
from .memory import MemoryStore, get_memory_store
from .planner import select_playbook
from .retrieval import retrieve_for_causal, retrieve_top_k
from .tools_causal_estimation import run_causal_estimation, select_estimator
from .tools_eda import (
    EDAToolError,
    correlation,
    dataset_overview,
    groupby_aggregate,
    time_trend,
    univariate_summary,
)


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

    # Playbook (Phase 2)
    playbook: str
    playbook_params: dict
    playbook_results: list[dict]  # Results from playbook execution

    # Causal (Phase 3)
    causal_spec_override: dict | None  # User-provided causal specification
    causal_readiness_status: str  # PASS/WARN/FAIL
    causal_report: dict | None  # CausalReadinessReport as dict

    # Causal Estimation (Phase 4)
    causal_confirmations: dict | None  # CausalConfirmations as dict
    confirmations_ok: bool  # True if confirmations valid and ok_to_estimate=True
    causal_estimate: dict | None  # CausalEstimateArtifact as dict

    # Memory (Phase 6)
    conversation_history: list[dict]  # Recent ConversationTurns as dicts

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

    # Check for causal patterns (Phase 3: route to CAUSAL for gate processing)
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, question_lower):
            route = "CAUSAL"
            reasons = [f"Matched causal pattern: {pattern}", "Causal analysis will run readiness gate"]
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


def load_memory_node(state: AgentState, memory_store: MemoryStore) -> dict:
    """Load conversation history from memory store."""
    session_id = state.get("session_id")
    trace_events = state.get("trace_events", [])

    if not session_id:
        # No session, no memory to load
        trace_events = trace_events + [{
            "event_type": "MEMORY_SKIPPED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"reason": "no_session_id"},
        }]
        return {"conversation_history": [], "trace_events": trace_events}

    # Load recent turns
    turns = memory_store.get_recent_turns(session_id, limit=10)
    history = [
        {
            "role": t.role,
            "content": t.content,
            "route": t.route,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
        }
        for t in turns
    ]

    trace_events = trace_events + [{
        "event_type": "MEMORY_LOADED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {"session_id": session_id, "num_turns": len(history)},
    }]

    return {"conversation_history": history, "trace_events": trace_events}


def save_memory_node(state: AgentState, memory_store: MemoryStore) -> dict:
    """Save the current turn to memory store."""
    session_id = state.get("session_id")
    trace_events = state.get("trace_events", [])

    if not session_id:
        return {"trace_events": trace_events}

    # Save user turn
    memory_store.add_turn(
        session_id=session_id,
        role="user",
        content=state["question"],
        route=state.get("route"),
    )

    # Save assistant turn
    answer = state.get("llm_response", "")
    artifacts = state.get("artifacts", [])
    artifacts_summary = [a.get("type", "unknown") for a in artifacts[:5]]

    memory_store.add_turn(
        session_id=session_id,
        role="assistant",
        content=answer,
        route=state.get("route"),
        artifacts_summary=artifacts_summary,
    )

    trace_events = trace_events + [{
        "event_type": "MEMORY_SAVED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {"session_id": session_id, "turns_added": 2},
    }]

    return {"trace_events": trace_events}


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


def retrieve_causal_context_node(
    state: AgentState,
    embeddings_client: EmbeddingsClient,
    storage_dir: Path,
) -> dict:
    """
    Retrieve context specifically for causal analysis.

    Uses specialized retrieval that prioritizes:
    - Known Caveats section (potential confounders, biases)
    - Causal Assumptions section
    - Data Dictionary (variable definitions for confounders)

    This ensures the causal gate has access to documented assumptions.
    """
    trace_events = state.get("trace_events", [])

    try:
        chunks = retrieve_for_causal(
            doc_id=state["doc_id"],
            query=state["question"],
            embeddings_client=embeddings_client,
            storage_dir=storage_dir,
        )

        # Serialize chunks to dict
        retrieved = [
            {
                "chunk_index": c.chunk_index,
                "text": c.text,
                "score": c.score,
                "section": c.section,
            }
            for c in chunks
        ]

        # Track which sections were found
        sections_found = list({c.section for c in chunks if c.section})

        trace_events = trace_events + [{
            "event_type": "CAUSAL_CONTEXT_RETRIEVED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "num_chunks": len(chunks),
                "chunk_indices": [c.chunk_index for c in chunks],
                "scores": [c.score for c in chunks],
                "sections_found": sections_found,
            },
        }]

        return {"retrieved_chunks": retrieved, "trace_events": trace_events}

    except FileNotFoundError:
        trace_events = trace_events + [{
            "event_type": "CAUSAL_CONTEXT_RETRIEVED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": "embeddings_not_found"},
        }]
        return {"retrieved_chunks": [], "trace_events": trace_events}


def compose_prompt_node(state: AgentState) -> dict:
    """Compose the prompt for the LLM with retrieved context, history, and playbook results."""
    system_instruction = (
        "You are a helpful data analysis assistant. "
        "Answer the user's question using the provided context and analysis results. "
        "Reference specific numbers and findings from the analysis. "
        "Be concise and accurate."
    )

    # Format conversation history (Phase 6)
    history_parts = []
    for turn in state.get("conversation_history", []):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if role == "user":
            history_parts.append(f"User: {content}")
        else:
            # Truncate long assistant responses in history
            truncated = content[:500] + "..." if len(content) > 500 else content
            history_parts.append(f"Assistant: {truncated}")

    history_text = "\n".join(history_parts) if history_parts else ""

    # Format retrieved chunks (context document)
    context_parts = []
    for chunk in state.get("retrieved_chunks", []):
        context_parts.append(f"[Chunk {chunk['chunk_index']}]: {chunk['text']}")

    context_text = "\n\n".join(context_parts) if context_parts else "(No context document available)"

    # Format playbook results
    analysis_parts = []
    for result in state.get("playbook_results", []):
        # Format text artifact
        if "text_artifact" in result:
            analysis_parts.append(result["text_artifact"].get("content", ""))
        # Format table artifact
        if "table_artifact" in result:
            table = result["table_artifact"]
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            if headers and rows:
                table_str = " | ".join(headers) + "\n"
                table_str += "-" * 40 + "\n"
                for row in rows[:20]:  # Limit rows in prompt
                    table_str += " | ".join(str(cell) for cell in row) + "\n"
                if len(rows) > 20:
                    table_str += f"... and {len(rows) - 20} more rows\n"
                analysis_parts.append(table_str)

    analysis_text = "\n\n".join(analysis_parts) if analysis_parts else "(No analysis results)"

    # Build prompt with optional history section
    if history_text:
        prompt = f"""{system_instruction}

PREVIOUS CONVERSATION:
{history_text}

CONTEXT (definitions and documentation):
{context_text}

ANALYSIS RESULTS:
{analysis_text}

User question: {state["question"]}

Answer:"""
    else:
        prompt = f"""{system_instruction}

CONTEXT (definitions and documentation):
{context_text}

ANALYSIS RESULTS:
{analysis_text}

User question: {state["question"]}

Answer:"""

    return {"prompt": prompt}


def call_llm_node(
    state: AgentState,
    llm_client: LLMClient,
    provider_info: LLMProviderInfo | None = None,
) -> dict:
    """
    Call the LLM with the composed prompt.

    Phase 5: Added provider tracking and shadow mode support.
    """
    trace_events = list(state.get("trace_events", []))
    prompt = state.get("prompt", "")

    # Add LLM_PROVIDER_USED trace event
    if provider_info:
        trace_events.append(create_provider_trace_event(provider_info))

    try:
        response = llm_client.generate(prompt)
        trace_events.append({
            "event_type": "LLM_CALLED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "prompt_length": len(prompt),
                "response_length": len(response),
            },
        })

        # Shadow mode: run secondary LLM call if enabled
        shadow_client, shadow_model = get_shadow_client()
        if shadow_client is not None:
            shadow_result = run_shadow_call(
                client=shadow_client,
                prompt=prompt,
                model=shadow_model,
            )

            # Add shadow result trace event
            trace_events.append(create_shadow_result_event(shadow_result))

            # Compute and add diff trace event
            diff = compute_shadow_diff(response, shadow_result.answer_text)
            trace_events.append(create_shadow_diff_event(diff))

        return {"llm_response": response, "trace_events": trace_events}

    except Exception as e:
        trace_events.append({
            "event_type": "LLM_CALLED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": str(e)},
        })
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


def select_playbook_node(state: AgentState, datasets_dir: Path) -> dict:
    """Select appropriate playbook based on question and dataset metadata."""
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])

    dataset_id = state.get("dataset_id")
    if not dataset_id:
        # No dataset - skip playbook selection
        trace_events = trace_events + [{
            "event_type": "PLAYBOOK_SELECTED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"playbook": "NONE", "reason": "no_dataset_id"},
        }]
        return {"playbook": "NONE", "playbook_params": {}, "trace_events": trace_events}

    # Load dataset metadata
    metadata_path = datasets_dir / dataset_id / "metadata.json"
    if not metadata_path.exists():
        trace_events = trace_events + [{
            "event_type": "PLAYBOOK_SELECTED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"playbook": "NONE", "reason": "dataset_not_found"},
        }]
        return {"playbook": "NONE", "playbook_params": {}, "trace_events": trace_events}

    with open(metadata_path) as f:
        metadata = json.load(f)

    column_names = metadata.get("column_names", [])
    inferred_types = metadata.get("inferred_types", {})

    # Use the planner
    selection = select_playbook(state["question"], column_names, inferred_types)

    trace_events = trace_events + [{
        "event_type": "PLAYBOOK_SELECTED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "playbook": selection.playbook,
            "confidence": selection.confidence,
            "params": selection.params,
        },
    }]

    # If needs clarification, add to artifacts
    if selection.playbook == "NEEDS_CLARIFICATION" and selection.clarification_questions:
        artifacts = artifacts + [{
            "type": "checklist",
            "items": selection.clarification_questions,
        }]

    return {
        "playbook": selection.playbook,
        "playbook_params": selection.params,
        "trace_events": trace_events,
        "artifacts": artifacts,
    }


def execute_playbook_node(state: AgentState, datasets_dir: Path) -> dict:
    """Execute the selected playbook and generate artifacts."""
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])
    playbook_results = []

    playbook = state.get("playbook", "NONE")
    params = state.get("playbook_params", {})
    dataset_id = state.get("dataset_id")

    if playbook == "NONE" or not dataset_id:
        return {"playbook_results": [], "trace_events": trace_events}

    try:
        if playbook == "OVERVIEW":
            result = dataset_overview(dataset_id, datasets_dir)
            artifacts = artifacts + [result["text_artifact"], result["table_artifact"]]
            playbook_results.append(result)

        elif playbook == "UNIVARIATE":
            column = params.get("column")
            if column:
                result = univariate_summary(dataset_id, column, datasets_dir)
                artifacts = artifacts + [result["table_artifact"]]
                playbook_results.append(result)

        elif playbook == "GROUPBY":
            group_col = params.get("group_col")
            target_col = params.get("target_col")
            agg = params.get("agg", "sum")
            if group_col and target_col:
                result = groupby_aggregate(dataset_id, group_col, target_col, agg, datasets_dir)
                artifacts = artifacts + [result["table_artifact"]]
                playbook_results.append(result)

        elif playbook == "TREND":
            date_col = params.get("date_col")
            target_col = params.get("target_col")
            agg = params.get("agg", "sum")
            freq = params.get("freq", "D")
            if date_col and target_col:
                result = time_trend(dataset_id, date_col, target_col, agg, freq, datasets_dir)
                artifacts = artifacts + [result["table_artifact"]]
                playbook_results.append(result)

        elif playbook == "CORRELATION":
            columns = params.get("columns", [])
            if len(columns) >= 2:
                result = correlation(dataset_id, columns, datasets_dir)
                artifacts = artifacts + [result["table_artifact"]]
                playbook_results.append(result)

        trace_events = trace_events + [{
            "event_type": "TOOLS_EXECUTED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "playbook": playbook,
                "num_results": len(playbook_results),
            },
        }]

    except EDAToolError as e:
        trace_events = trace_events + [{
            "event_type": "TOOLS_EXECUTED",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": str(e)},
        }]
        artifacts = artifacts + [{
            "type": "text",
            "content": f"Error executing playbook: {e}",
        }]

    return {
        "playbook_results": playbook_results,
        "artifacts": artifacts,
        "trace_events": trace_events,
    }


def causal_gate_node(state: AgentState, datasets_dir: Path) -> dict:
    """
    Run the causal readiness gate and return structured report.

    Phase 3: Never returns ATE - only readiness assessment and diagnostics.

    Now uses retrieved context to inform assumptions and confounder identification.
    """
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])

    dataset_id = state.get("dataset_id")
    doc_id = state.get("doc_id", "")
    question = state.get("question", "")
    spec_override = state.get("causal_spec_override")

    # Get retrieved context chunks (added in retrieve_causal_context_node)
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Load dataset metadata if available
    column_names: list[str] = []
    inferred_types: dict[str, str] = {}

    if dataset_id:
        metadata_path = datasets_dir / dataset_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            column_names = metadata.get("column_names", [])
            inferred_types = metadata.get("inferred_types", {})

    # Run the causal readiness gate
    report = causal_readiness_gate(
        question=question,
        doc_id=doc_id,
        dataset_id=dataset_id,
        column_names=column_names,
        inferred_types=inferred_types,
        datasets_dir=datasets_dir,
        spec_override=spec_override,
        # Pass context for future use (could inform confounder inference)
        # context_text=causal_context,  # Reserved for future enhancement
    )

    # Convert report to dict for serialization
    report_dict = report.model_dump()

    # Add retrieved causal context as an artifact for transparency
    if retrieved_chunks:
        context_summary = []
        for chunk in retrieved_chunks[:5]:  # Limit to top 5
            section = chunk.get("section", "Unknown")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")[:150]
            context_summary.append(f"[{section}] (score: {score:.3f}): {text}...")

        artifacts.append({
            "type": "text",
            "content": "📚 Retrieved causal context:\n" + "\n".join(context_summary),
        })

    # Add diagnostics as artifacts
    for diag in report.diagnostics:
        artifacts.append(diag.model_dump())

    # Add the spec artifact
    artifacts.append(report.spec.model_dump())

    # Add the readiness report artifact
    artifacts.append(report_dict)

    trace_events = trace_events + [{
        "event_type": "CAUSAL_GATE_EXECUTED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "readiness_status": report.readiness_status,
            "n_diagnostics": len(report.diagnostics),
            "ready_for_estimation": report.ready_for_estimation,
            "context_chunks_used": len(retrieved_chunks),
        },
    }]

    return {
        "causal_readiness_status": report.readiness_status,
        "causal_report": report_dict,
        "artifacts": artifacts,
        "trace_events": trace_events,
    }


def causal_decide_next_node(state: AgentState) -> dict:
    """
    Decide next action after causal gate.

    - FAIL -> return NEEDS_CLARIFICATION with followup questions
    - WARN -> return NEEDS_CLARIFICATION with targeted questions
    - PASS -> return "Ready for estimation" message (no ATE in Phase 3)
    """
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])

    readiness_status = state.get("causal_readiness_status", "FAIL")
    causal_report = state.get("causal_report", {})
    followup_questions = causal_report.get("followup_questions", [])
    recommended_estimators = causal_report.get("recommended_estimators", [])

    if readiness_status == "FAIL":
        # Add checklist of required clarifications
        if followup_questions:
            artifacts.append({
                "type": "checklist",
                "items": followup_questions,
            })

        artifacts.append({
            "type": "text",
            "content": "❌ Causal readiness: FAIL\n\nCausal estimation cannot proceed until the above issues are resolved.",
        })

        return {
            "route": "NEEDS_CLARIFICATION",
            "llm_response": "Causal analysis requires clarification before proceeding. Please address the items in the checklist.",
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    elif readiness_status == "WARN":
        # Add targeted questions
        if followup_questions:
            artifacts.append({
                "type": "checklist",
                "items": followup_questions,
            })

        artifacts.append({
            "type": "text",
            "content": "⚠️ Causal readiness: WARN\n\nCausal estimation may proceed with caveats. Please review the warnings and confirm assumptions.",
        })

        return {
            "route": "NEEDS_CLARIFICATION",
            "llm_response": "Causal analysis has warnings that should be addressed. Please review the diagnostics and confirm assumptions.",
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    else:  # PASS
        # Ready for estimation - but Phase 3 does NOT compute ATE
        spec = causal_report.get("spec", {})
        treatment = spec.get("treatment", "unknown")
        outcome = spec.get("outcome", "unknown")
        confounders = spec.get("confounders_selected", [])

        message = f"""✅ Causal readiness: PASS

**Ready for estimation** (Phase 4 will compute actual effects)

**Causal Specification:**
- Treatment: `{treatment}`
- Outcome: `{outcome}`
- Confounders: {confounders if confounders else 'None selected'}

**Recommended Estimators:**
{chr(10).join('- ' + e for e in recommended_estimators) if recommended_estimators else '- To be determined'}

**Next Steps (Phase 4):**
1. Select estimation method
2. Compute propensity scores / matching
3. Estimate Average Treatment Effect (ATE)
4. Sensitivity analysis

*Note: Actual effect estimation is deferred to Phase 4.*"""

        artifacts.append({
            "type": "text",
            "content": message,
        })

        return {
            "llm_response": message,
            "artifacts": artifacts,
            "trace_events": trace_events,
            "confirmations_ok": False,  # Need confirmations before estimation
        }


def check_confirmations_node(state: AgentState) -> dict:
    """
    Check if user has provided required confirmations for causal estimation.

    Phase 4: Estimation only proceeds if:
    1. causal_readiness_status is PASS or WARN (not FAIL)
    2. causal_confirmations is provided
    3. causal_confirmations.ok_to_estimate == True
    4. Treatment is binary (already checked in gate)
    """
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])

    confirmations = state.get("causal_confirmations")
    readiness_status = state.get("causal_readiness_status", "FAIL")

    trace_events = trace_events + [{
        "event_type": "CONFIRMATIONS_CHECKED",
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "confirmations_present": confirmations is not None,
            "readiness_status": readiness_status,
        },
    }]

    # If readiness is FAIL, cannot estimate (WARN can proceed with confirmations)
    if readiness_status == "FAIL":
        return {
            "confirmations_ok": False,
            "trace_events": trace_events,
        }

    # If no confirmations provided, ask for them
    if not confirmations:
        missing_questions = [
            "To proceed with causal estimation, please confirm:",
            "1. What is the assignment mechanism? (randomized, policy, self_selected, unknown)",
            "2. Is there potential for interference between units? (no_interference, possible_interference, unknown)",
            "3. What is your missing data policy? (listwise_delete, simple_impute, unknown)",
            "4. Do you authorize estimation to proceed? (ok_to_estimate: true/false)",
        ]
        artifacts.append({
            "type": "checklist",
            "items": missing_questions,
        })
        artifacts.append({
            "type": "text",
            "content": "⚠️ **Confirmations Required**\n\nPlease provide causal_confirmations in your request to proceed with estimation.",
        })
        return {
            "confirmations_ok": False,
            "route": "NEEDS_CLARIFICATION",
            "llm_response": "Please provide confirmations to proceed with causal estimation.",
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    # Check if ok_to_estimate is True
    ok_to_estimate = confirmations.get("ok_to_estimate", False)
    if not ok_to_estimate:
        artifacts.append({
            "type": "text",
            "content": "❌ **Estimation Declined**\n\nUser set ok_to_estimate=False. Estimation will not proceed.",
        })
        return {
            "confirmations_ok": False,
            "route": "NEEDS_CLARIFICATION",
            "llm_response": "Estimation declined by user (ok_to_estimate=False).",
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    # All checks passed
    return {
        "confirmations_ok": True,
        "trace_events": trace_events,
    }


def run_estimation_node(state: AgentState, datasets_dir: Path) -> dict:
    """
    Run causal estimation if confirmations are OK.

    Phase 4: Produces CausalEstimateArtifact with ATE and CI.
    """
    trace_events = state.get("trace_events", [])
    artifacts = state.get("artifacts", [])

    confirmations_ok = state.get("confirmations_ok", False)
    if not confirmations_ok:
        return {"trace_events": trace_events}

    # Get causal spec from report
    causal_report = state.get("causal_report", {})
    spec = causal_report.get("spec", {})
    treatment = spec.get("treatment")
    outcome = spec.get("outcome")
    confounders = spec.get("confounders_selected", [])
    dataset_id = state.get("dataset_id")

    if not all([treatment, outcome, dataset_id]):
        artifacts.append({
            "type": "text",
            "content": "❌ Cannot run estimation: missing treatment, outcome, or dataset.",
        })
        return {
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    # Get diagnostic status to select estimator
    diagnostics = causal_report.get("diagnostics", [])
    positivity_status = "PASS"
    for diag in diagnostics:
        if diag.get("name") == "positivity_check":
            positivity_status = diag.get("status", "PASS")
            break

    # Select estimator
    method = select_estimator(positivity_status, len(confounders))

    try:
        result = run_causal_estimation(
            dataset_id=dataset_id,
            treatment_col=treatment,
            outcome_col=outcome,
            covariates=confounders,
            datasets_dir=datasets_dir,
            method=method,
        )

        estimate_artifact = result.get("estimate_artifact", {})

        # Add estimate artifact
        artifacts.append(estimate_artifact)

        # Add propensity summary if IPW
        if "propensity_summary" in result:
            artifacts.append(result["propensity_summary"])

        trace_events = trace_events + [{
            "event_type": "ESTIMATION_RAN",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "method": method,
                "n_used": estimate_artifact.get("n_used", 0),
                "estimate": estimate_artifact.get("estimate"),
                "ci_low": estimate_artifact.get("ci_low"),
                "ci_high": estimate_artifact.get("ci_high"),
            },
        }]

        # Build summary text
        ate = estimate_artifact.get("estimate", 0)
        ci_low = estimate_artifact.get("ci_low", 0)
        ci_high = estimate_artifact.get("ci_high", 0)
        n_used = estimate_artifact.get("n_used", 0)
        warnings = estimate_artifact.get("warnings", [])

        summary = f"""✅ **Causal Effect Estimate**

**Method:** {method.replace('_', ' ').title()}
**Estimand:** Average Treatment Effect (ATE)

**Result:**
- **ATE = {ate:.4f}** (95% CI: [{ci_low:.4f}, {ci_high:.4f}])
- Observations used: {n_used}
- Covariates adjusted: {confounders if confounders else 'None'}

**Interpretation:**
On average, receiving treatment is associated with a {abs(ate):.4f} {'increase' if ate > 0 else 'decrease'} in the outcome, holding confounders constant.
"""
        if warnings:
            summary += f"\n**Warnings:** {', '.join(warnings)}"

        artifacts.append({
            "type": "text",
            "content": summary,
        })

        return {
            "causal_estimate": estimate_artifact,
            "llm_response": summary,
            "artifacts": artifacts,
            "trace_events": trace_events,
        }

    except Exception as e:
        trace_events = trace_events + [{
            "event_type": "ESTIMATION_RAN",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"error": str(e)},
        }]
        artifacts.append({
            "type": "text",
            "content": f"❌ Estimation failed: {e}",
        })
        return {
            "artifacts": artifacts,
            "trace_events": trace_events,
        }


def create_agent_graph(
    embeddings_client: EmbeddingsClient,
    llm_client: LLMClient,
    storage_dir: Path,
    datasets_dir: Path | None = None,
    provider_info: LLMProviderInfo | None = None,
    memory_store: MemoryStore | None = None,
) -> StateGraph:
    """
    Create the LangGraph agent graph.

    Args:
        embeddings_client: Client for embeddings
        llm_client: Client for LLM generation
        storage_dir: Path to document storage
        datasets_dir: Path to dataset storage (for playbooks)
        provider_info: LLM provider information for tracing (Phase 5)
        memory_store: Session memory store (Phase 6)

    Returns:
        Compiled StateGraph
    """
    # Default datasets_dir if not provided
    if datasets_dir is None:
        datasets_dir = storage_dir.parent / "datasets"

    # Default memory store if not provided
    if memory_store is None:
        memory_store = get_memory_store()

    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Phase 6: Memory nodes
    graph.add_node("load_memory", lambda s: load_memory_node(s, memory_store))
    graph.add_node("save_memory", lambda s: save_memory_node(s, memory_store))

    # Add nodes (note: node names must not conflict with state keys)
    graph.add_node("router", route_node)
    graph.add_node("retrieve_context", lambda s: retrieve_context_node(s, embeddings_client, storage_dir))
    graph.add_node("select_playbook", lambda s: select_playbook_node(s, datasets_dir))
    graph.add_node("execute_playbook", lambda s: execute_playbook_node(s, datasets_dir))
    graph.add_node("compose_prompt", compose_prompt_node)
    graph.add_node("call_llm", lambda s: call_llm_node(s, llm_client, provider_info))
    graph.add_node("build_response", build_response_node)

    # Phase 3: Causal gate nodes
    # NEW: Retrieve causal context before running the gate
    graph.add_node("retrieve_causal_context", lambda s: retrieve_causal_context_node(s, embeddings_client, storage_dir))
    graph.add_node("causal_gate", lambda s: causal_gate_node(s, datasets_dir))
    graph.add_node("causal_decide_next", causal_decide_next_node)

    # Phase 4: Causal estimation nodes
    graph.add_node("check_confirmations", check_confirmations_node)
    graph.add_node("run_estimation", lambda s: run_estimation_node(s, datasets_dir))

    # Define edges - Phase 6: start with load_memory
    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "router")

    # Conditional routing based on route type
    def route_after_router(state: AgentState) -> str:
        route = state.get("route", "ANALYSIS")
        if route == "CAUSAL":
            # NEW: Route to causal context retrieval first
            return "retrieve_causal_context"
        elif route in ["NEEDS_CLARIFICATION", "REPORTING"]:
            return "build_response"
        return "retrieve_context"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_causal_context": "retrieve_causal_context",  # NEW
            "retrieve_context": "retrieve_context",
            "build_response": "build_response",
        }
    )

    # Causal flow: retrieve context -> gate -> decide -> check confirmations -> estimation -> response
    graph.add_edge("retrieve_causal_context", "causal_gate")  # NEW
    graph.add_edge("causal_gate", "causal_decide_next")

    # After causal_decide_next, check if we should proceed to estimation
    # PASS and WARN can proceed to confirmations; FAIL cannot
    def route_after_causal_decide(state: AgentState) -> str:
        readiness = state.get("causal_readiness_status", "FAIL")
        if readiness in ["PASS", "WARN"]:
            return "check_confirmations"
        return "build_response"

    graph.add_conditional_edges(
        "causal_decide_next",
        route_after_causal_decide,
        {
            "check_confirmations": "check_confirmations",
            "build_response": "build_response",
        }
    )

    # After check_confirmations, run estimation if OK
    def route_after_confirmations(state: AgentState) -> str:
        if state.get("confirmations_ok", False):
            return "run_estimation"
        return "build_response"

    graph.add_conditional_edges(
        "check_confirmations",
        route_after_confirmations,
        {
            "run_estimation": "run_estimation",
            "build_response": "build_response",
        }
    )

    graph.add_edge("run_estimation", "build_response")

    # After retrieval, select and execute playbook, then compose prompt
    graph.add_edge("retrieve_context", "select_playbook")
    graph.add_edge("select_playbook", "execute_playbook")
    graph.add_edge("execute_playbook", "compose_prompt")
    graph.add_edge("compose_prompt", "call_llm")
    graph.add_edge("call_llm", "build_response")

    # Phase 6: save memory after response, then end
    graph.add_edge("build_response", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


def run_agent(
    question: str,
    doc_id: str,
    embeddings_client: EmbeddingsClient,
    llm_client: LLMClient,
    storage_dir: Path,
    dataset_id: str | None = None,
    session_id: str | None = None,
    datasets_dir: Path | None = None,
    causal_spec_override: CausalSpecOverride | dict | None = None,
    causal_confirmations: CausalConfirmations | dict | None = None,
    provider_info: LLMProviderInfo | None = None,
    memory_store: MemoryStore | None = None,
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
        datasets_dir: Optional datasets directory path
        causal_spec_override: Optional causal specification override (Phase 3)
        causal_confirmations: Optional causal confirmations for estimation (Phase 4)
        provider_info: LLM provider information for tracing (Phase 5)
        memory_store: Session memory store (Phase 6)

    Returns:
        Final AgentState (dict) with response
    """
    import uuid as uuid_module

    graph = create_agent_graph(
        embeddings_client,
        llm_client,
        storage_dir,
        datasets_dir,
        provider_info,
        memory_store,
    )

    # Convert CausalSpecOverride to dict if needed
    spec_override_dict = None
    if causal_spec_override is not None:
        if isinstance(causal_spec_override, dict):
            spec_override_dict = causal_spec_override
        else:
            spec_override_dict = causal_spec_override.model_dump()

    # Convert CausalConfirmations to dict if needed
    confirmations_dict = None
    if causal_confirmations is not None:
        if isinstance(causal_confirmations, dict):
            confirmations_dict = causal_confirmations
        else:
            confirmations_dict = causal_confirmations.model_dump()

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
        "playbook": "",
        "playbook_params": {},
        "playbook_results": [],
        "causal_spec_override": spec_override_dict,
        "causal_readiness_status": "",
        "causal_report": None,
        "causal_confirmations": confirmations_dict,
        "confirmations_ok": False,
        "causal_estimate": None,
        "conversation_history": [],  # Phase 6: loaded by load_memory_node
    }

    final_state = graph.invoke(initial_state)
    return final_state

