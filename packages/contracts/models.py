"""
Pydantic models for all API contracts.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel


class UploadContextDocResponse(BaseModel):
    """Response from uploading a context document."""
    doc_id: str
    doc_hash: str
    num_chars: int
    num_chunks: int
    status: Literal["indexed", "failed"]
    errors: Optional[list[str]] = None


class UploadDatasetResponse(BaseModel):
    """Response from uploading a dataset."""
    dataset_id: str
    dataset_hash: str
    n_rows: int
    n_cols: int
    column_names: list[str]
    inferred_types: dict[str, str]
    status: Literal["profiled", "failed"]
    errors: Optional[list[str]] = None


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


# Discriminated union for artifacts
Artifact = Union[TextArtifact, TableArtifact, ChecklistArtifact]


class TraceEvent(BaseModel):
    """Trace event for observability."""
    event_type: str
    timestamp: str
    payload: dict


class AskQuestionRequest(BaseModel):
    """Request to ask a question."""
    question: str
    doc_id: str
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None


class AskQuestionResponse(BaseModel):
    """Response from asking a question."""
    answer_text: str
    router_decision: RouterDecision
    artifacts: list[Artifact]
    trace_id: str

