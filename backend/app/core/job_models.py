from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class JobState(BaseModel):
    job_id: str
    status: str = "CREATED"  # CREATED, QUEUED, RUNNING, CLARIFY, COMPLETED, FAILED
    progress: int = 0
    stage: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: Optional[str] = None

    def update_status(
        self, status: str, stage: Optional[str] = None, progress: Optional[int] = None
    ):
        self.status = status
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = int(progress)
        self.updated_at = datetime.now(timezone.utc).isoformat()
