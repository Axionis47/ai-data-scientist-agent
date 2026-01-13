"""
Memory layer for session management.

Provides:
- FileSessionStore: File-based storage for dev/test (no external deps)
- get_memory_store(): Factory that returns appropriate store for environment

File-based storage works for single-instance dev. For production with
multiple Cloud Run instances, migrate to Redis or Firestore.
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from packages.contracts.models import (
    ConversationTurn,
    MemoryStore,
    SessionMemory,
)

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_MEMORY_DIR = Path("storage/sessions")


class FileSessionStore:
    """
    File-based session store for development and testing.

    Each session is stored as a JSON file:
        storage/sessions/{session_id}.json

    Thread-safe for single process (uses atomic file writes).
    NOT suitable for multi-instance production (use Redis/Firestore).
    """

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir or DEFAULT_MEMORY_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.storage_dir / f"{safe_id}.json"

    def _load_session(self, session_id: str) -> SessionMemory | None:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return SessionMemory.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def _save_session(self, session: SessionMemory) -> None:
        """Save a session to disk (atomic write)."""
        path = self._session_path(session.session_id)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(session.model_dump(mode="json"), f, indent=2, default=str)
        temp_path.rename(path)

    def get_session(self, session_id: str) -> SessionMemory | None:
        """Get a session by ID. Returns None if not found."""
        return self._load_session(session_id)

    def create_session(
        self,
        session_id: str,
        doc_id: str | None = None,
        dataset_id: str | None = None,
    ) -> SessionMemory:
        """Create a new session."""
        now = datetime.now(UTC)
        session = SessionMemory(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            doc_id=doc_id,
            dataset_id=dataset_id,
            turns=[],
            user_context={},
        )
        self._save_session(session)
        logger.info(f"Created session {session_id}")
        return session

    def add_turn(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        route: str | None = None,
        artifacts_summary: list[str] | None = None,
    ) -> ConversationTurn:
        """Add a turn to an existing session. Creates session if needed."""
        session = self._load_session(session_id)
        if session is None:
            session = self.create_session(session_id)

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            role=role,
            content=content,
            route=route,
            artifacts_summary=artifacts_summary or [],
        )
        session.turns.append(turn)
        session.updated_at = datetime.now(UTC)
        self._save_session(session)
        return turn

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[ConversationTurn]:
        """Get the most recent turns from a session."""
        session = self._load_session(session_id)
        if session is None:
            return []
        return session.turns[-limit:]

    def update_session_context(
        self,
        session_id: str,
        context_updates: dict,
    ) -> SessionMemory:
        """Update user_context for a session (merge with existing)."""
        session = self._load_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        session.user_context.update(context_updates)
        session.updated_at = datetime.now(UTC)
        self._save_session(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all session IDs (for debugging/admin)."""
        return [
            p.stem for p in self.storage_dir.glob("*.json")
            if not p.name.endswith(".tmp")
        ]


# Type check: ensure FileSessionStore implements MemoryStore protocol
def _verify_protocol() -> None:
    """Verify FileSessionStore implements MemoryStore at import time."""
    store: MemoryStore = FileSessionStore()
    assert isinstance(store, MemoryStore), "FileSessionStore must implement MemoryStore"


_verify_protocol()


def get_memory_store(storage_dir: Path | None = None) -> MemoryStore:
    """
    Factory function to get the appropriate memory store.

    For now, always returns FileSessionStore.
    In production, this would check APP_ENV and return Redis/Firestore client.

    Args:
        storage_dir: Optional override for storage directory

    Returns:
        MemoryStore implementation
    """
    # Future: check os.getenv("APP_ENV") and return Redis/Firestore for prod
    app_env = os.getenv("APP_ENV", "dev")

    if app_env in ("dev", "test"):
        return FileSessionStore(storage_dir=storage_dir)
    else:
        # TODO: Return Redis/Firestore store for staging/prod
        # For now, fall back to file-based (single instance OK for MVP)
        logger.warning(
            f"Using FileSessionStore in {app_env} environment. "
            "Consider migrating to Redis/Firestore for multi-instance."
        )
        return FileSessionStore(storage_dir=storage_dir)

