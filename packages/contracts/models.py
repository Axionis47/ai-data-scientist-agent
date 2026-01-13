"""
Pydantic models for all API contracts.
"""

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# =============================================================================
# Phase 6: Memory Layer Contracts
# =============================================================================


class ConversationTurn(BaseModel):
    """A single turn in a conversation (user question + agent response)."""
    turn_id: str  # UUID for this turn
    timestamp: datetime
    role: Literal["user", "assistant"]
    content: str  # The question or answer text
    # Optional metadata
    route: str | None = None  # ANALYSIS, CAUSAL, etc.
    artifacts_summary: list[str] = []  # Brief descriptions of artifacts produced
    dataset_id: str | None = None
    doc_id: str | None = None


class SessionMemory(BaseModel):
    """Memory for a single session (conversation history)."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    doc_id: str | None = None  # Context doc for this session
    dataset_id: str | None = None  # Dataset for this session
    turns: list[ConversationTurn] = []
    # Summary for long conversations (optional, for context window management)
    conversation_summary: str | None = None
    # User-provided context that persists across turns
    user_context: dict = Field(default_factory=dict)


class MemoryStats(BaseModel):
    """Statistics about memory usage."""
    total_sessions: int
    total_turns: int
    oldest_session: datetime | None = None
    newest_session: datetime | None = None


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for session memory storage backends."""

    def get_session(self, session_id: str) -> SessionMemory | None:
        """Get a session by ID. Returns None if not found."""
        ...

    def create_session(
        self,
        session_id: str,
        doc_id: str | None = None,
        dataset_id: str | None = None,
    ) -> SessionMemory:
        """Create a new session."""
        ...

    def add_turn(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        route: str | None = None,
        artifacts_summary: list[str] | None = None,
    ) -> ConversationTurn:
        """Add a turn to an existing session. Creates session if needed."""
        ...

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[ConversationTurn]:
        """Get the most recent turns from a session."""
        ...

    def update_session_context(
        self,
        session_id: str,
        context_updates: dict,
    ) -> SessionMemory:
        """Update user_context for a session (merge with existing)."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        ...


# =============================================================================
# Phase 4: Causal Confirmations and Estimation Artifacts
# =============================================================================


class CausalConfirmations(BaseModel):
    """User confirmations required before causal estimation can proceed."""
    assignment_mechanism: Literal["randomized", "policy", "self_selected", "unknown"] = "unknown"
    interference: Literal["no_interference", "possible_interference", "unknown"] = "unknown"
    missing_data_policy: Literal["listwise_delete", "simple_impute", "unknown"] = "unknown"
    ok_to_estimate: bool = False  # Must be True to produce ATE


class CausalEstimateArtifact(BaseModel):
    """Causal effect estimate artifact."""
    type: Literal["causal_estimate"] = "causal_estimate"
    method: str  # e.g., "regression_adjustment", "ipw"
    estimand: str  # e.g., "ATE", "ATT"
    estimate: float  # Point estimate
    ci_low: float  # 95% CI lower bound
    ci_high: float  # 95% CI upper bound
    n_used: int  # Number of observations used
    covariates: list[str]  # Covariates used in adjustment
    warnings: list[str] = []  # Any warnings about the estimate


# =============================================================================
# Phase 3: Causal Analysis Artifacts
# =============================================================================


class CausalSpecArtifact(BaseModel):
    """Causal specification artifact capturing treatment/outcome/confounders."""
    type: Literal["causal_spec"] = "causal_spec"
    treatment: str | None = None
    outcome: str | None = None
    unit: str | None = None  # Unit of observation (e.g., "customer_id")
    time_col: str | None = None  # Time column for ordering
    horizon: str | None = None  # Time horizon for outcome measurement
    confounders_selected: list[str] = []
    confounders_missing: list[str] = []  # Confounders we can't observe
    assumptions: list[str] = []  # Assumptions being made
    questions: list[str] = []  # Follow-up questions for user


class DiagnosticArtifact(BaseModel):
    """Diagnostic check result artifact."""
    type: Literal["diagnostic"] = "diagnostic"
    name: str  # e.g., "positivity_check", "missingness_check"
    status: Literal["PASS", "WARN", "FAIL"]
    details: dict = {}  # Diagnostic-specific details
    recommendations: list[str] = []  # What to do if WARN/FAIL


class CausalReadinessReport(BaseModel):
    """Overall causal readiness report artifact."""
    type: Literal["causal_readiness"] = "causal_readiness"
    readiness_status: Literal["PASS", "WARN", "FAIL"]
    spec: CausalSpecArtifact
    diagnostics: list[DiagnosticArtifact] = []
    followup_questions: list[str] = []
    ready_for_estimation: bool = False
    recommended_estimators: list[str] = []  # Phase 4: what estimators would be used


# =============================================================================
# Phase 0-2: Original Contracts (unchanged)
# =============================================================================


class UploadContextDocResponse(BaseModel):
    """Response from uploading a context document."""
    doc_id: str
    doc_hash: str
    num_chars: int
    num_chunks: int
    status: Literal["indexed", "failed"]
    errors: list[str] | None = None


class UploadDatasetResponse(BaseModel):
    """Response from uploading a dataset."""
    dataset_id: str
    dataset_hash: str
    n_rows: int
    n_cols: int
    column_names: list[str]
    inferred_types: dict[str, str]
    status: Literal["profiled", "failed"]
    errors: list[str] | None = None


class RouterDecision(BaseModel):
    """Router decision for query routing."""
    route: Literal["ANALYSIS", "CAUSAL", "REPORTING", "SYSTEM", "NEEDS_CLARIFICATION"]
    confidence: float
    reasons: list[str]


class TextArtifact(BaseModel):
    """Text artifact output."""
    type: Literal["text"] = "text"
    content: str


class TableArtifact(BaseModel):
    """Table artifact output."""
    type: Literal["table"] = "table"
    headers: list[str]
    rows: list[list[str]]


class ChecklistArtifact(BaseModel):
    """Checklist artifact output."""
    type: Literal["checklist"] = "checklist"
    items: list[str]


# Discriminated union for artifacts (Phase 4: added causal estimate)
Artifact = (
    TextArtifact
    | TableArtifact
    | ChecklistArtifact
    | CausalSpecArtifact
    | DiagnosticArtifact
    | CausalReadinessReport
    | CausalEstimateArtifact
)


class TraceEvent(BaseModel):
    """Trace event for observability."""
    event_type: str
    timestamp: str
    payload: dict


class CausalSpecOverride(BaseModel):
    """Optional user-provided causal specification override."""
    treatment: str | None = None
    outcome: str | None = None
    unit: str | None = None
    time_col: str | None = None
    horizon: str | None = None
    confounders: list[str] | None = None


class AskQuestionRequest(BaseModel):
    """Request to ask a question."""
    question: str
    doc_id: str
    dataset_id: str | None = None
    session_id: str | None = None
    # Phase 3: Optional causal specification override
    causal_spec_override: CausalSpecOverride | None = None
    # Phase 4: User confirmations for causal estimation
    causal_confirmations: CausalConfirmations | None = None


class VersionInfo(BaseModel):
    """Version information for reproducibility and tracking."""
    api_version: str = "0.1.0"
    prompt_version: str = "v1.0.0"
    embedding_model: str | None = None
    llm_model: str | None = None
    segment_versions: list[dict] = []  # Versions of retrieved chunks used


class AskQuestionResponse(BaseModel):
    """Response from asking a question."""
    answer_text: str
    router_decision: RouterDecision
    artifacts: list[Artifact]
    trace_id: str
    version_info: VersionInfo | None = None  # Added for segment/prompt versioning

