"""
Pydantic models for all API contracts.
"""

from typing import Literal

from pydantic import BaseModel

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


# Discriminated union for artifacts (Phase 3: added causal artifacts)
Artifact = (
    TextArtifact
    | TableArtifact
    | ChecklistArtifact
    | CausalSpecArtifact
    | DiagnosticArtifact
    | CausalReadinessReport
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


class AskQuestionResponse(BaseModel):
    """Response from asking a question."""
    answer_text: str
    router_decision: RouterDecision
    artifacts: list[Artifact]
    trace_id: str

