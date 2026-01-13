"""
Contracts package - Pydantic models for the SDLC API.
All request/response models are defined here for type safety.
"""

from .models import (
    Artifact,
    AskQuestionRequest,
    AskQuestionResponse,
    CausalConfirmations,
    CausalEstimateArtifact,
    CausalReadinessReport,
    CausalSpecArtifact,
    CausalSpecOverride,
    ChecklistArtifact,
    DiagnosticArtifact,
    RouterDecision,
    TableArtifact,
    TextArtifact,
    TraceEvent,
    UploadContextDocResponse,
    UploadDatasetResponse,
    VersionInfo,
)

__all__ = [
    "UploadContextDocResponse",
    "UploadDatasetResponse",
    "RouterDecision",
    "Artifact",
    "TextArtifact",
    "TableArtifact",
    "ChecklistArtifact",
    "TraceEvent",
    "AskQuestionRequest",
    "AskQuestionResponse",
    # Phase 3: Causal artifacts
    "CausalSpecArtifact",
    "DiagnosticArtifact",
    "CausalReadinessReport",
    "CausalSpecOverride",
    # Phase 4: Causal estimation
    "CausalConfirmations",
    "CausalEstimateArtifact",
    # Versioning
    "VersionInfo",
]

