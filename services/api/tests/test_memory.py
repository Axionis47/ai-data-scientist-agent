"""
Tests for the memory layer (Phase 6).

Tests:
- FileSessionStore CRUD operations
- Conversation history loading in agent
- Memory persistence across turns
"""

import tempfile
from pathlib import Path

from packages.agent.memory import FileSessionStore
from packages.contracts.models import MemoryStore


class TestFileSessionStore:
    """Unit tests for FileSessionStore."""

    def test_store_implements_protocol(self):
        """FileSessionStore should implement MemoryStore protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            assert isinstance(store, MemoryStore)

    def test_create_session(self):
        """Should create a new session with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            session = store.create_session(
                session_id="test-session-1",
                doc_id="doc-123",
                dataset_id="dataset-456",
            )
            assert session.session_id == "test-session-1"
            assert session.doc_id == "doc-123"
            assert session.dataset_id == "dataset-456"
            assert session.turns == []
            assert session.created_at is not None
            assert session.updated_at is not None

    def test_get_session_returns_none_for_missing(self):
        """Should return None for non-existent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            result = store.get_session("nonexistent")
            assert result is None

    def test_get_session_returns_created_session(self):
        """Should retrieve a previously created session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            store.create_session("test-session-2")
            session = store.get_session("test-session-2")
            assert session is not None
            assert session.session_id == "test-session-2"

    def test_add_turn_creates_session_if_needed(self):
        """Should create session automatically when adding first turn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            turn = store.add_turn(
                session_id="auto-created",
                role="user",
                content="Hello, world!",
            )
            assert turn.role == "user"
            assert turn.content == "Hello, world!"
            assert turn.turn_id is not None

            session = store.get_session("auto-created")
            assert session is not None
            assert len(session.turns) == 1

    def test_add_turn_appends_to_existing(self):
        """Should append turns to existing session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            store.create_session("multi-turn")

            store.add_turn("multi-turn", "user", "Question 1")
            store.add_turn("multi-turn", "assistant", "Answer 1")
            store.add_turn("multi-turn", "user", "Question 2")

            session = store.get_session("multi-turn")
            assert len(session.turns) == 3
            assert session.turns[0].content == "Question 1"
            assert session.turns[1].role == "assistant"
            assert session.turns[2].content == "Question 2"

    def test_get_recent_turns_limits_results(self):
        """Should return only the most recent N turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))

            for i in range(10):
                store.add_turn("many-turns", "user", f"Message {i}")

            recent = store.get_recent_turns("many-turns", limit=3)
            assert len(recent) == 3
            assert recent[0].content == "Message 7"
            assert recent[2].content == "Message 9"

    def test_get_recent_turns_empty_for_missing_session(self):
        """Should return empty list for non-existent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            recent = store.get_recent_turns("nonexistent", limit=5)
            assert recent == []

    def test_update_session_context(self):
        """Should merge context updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            store.create_session("context-test")

            store.update_session_context("context-test", {"key1": "value1"})
            store.update_session_context("context-test", {"key2": "value2"})

            session = store.get_session("context-test")
            assert session.user_context == {"key1": "value1", "key2": "value2"}

    def test_delete_session(self):
        """Should delete session and return True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            store.create_session("to-delete")

            result = store.delete_session("to-delete")
            assert result is True
            assert store.get_session("to-delete") is None

    def test_delete_session_returns_false_for_missing(self):
        """Should return False when deleting non-existent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            result = store.delete_session("never-existed")
            assert result is False

    def test_list_sessions(self):
        """Should list all session IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(storage_dir=Path(tmpdir))
            store.create_session("session-a")
            store.create_session("session-b")
            store.create_session("session-c")

            sessions = store.list_sessions()
            assert set(sessions) == {"session-a", "session-b", "session-c"}


def _make_fake_embedding(dim: int = 256) -> list[float]:
    """Create a fake embedding vector of the right dimension."""
    import random
    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(dim)]


class TestMemoryInAgentGraph:
    """Integration tests for memory in agent graph."""

    def test_agent_loads_conversation_history(self):
        """Agent should load conversation history from memory store."""
        import json

        from packages.agent.fake_clients import FakeEmbeddingsClient, FakeLLMClient
        from packages.agent.graph import run_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "contexts"
            storage_dir.mkdir()
            datasets_dir = Path(tmpdir) / "datasets"
            datasets_dir.mkdir()
            memory_dir = Path(tmpdir) / "sessions"
            memory_dir.mkdir()

            # Create a doc so retrieval works (must match FakeEmbeddingsClient dimension)
            doc_dir = storage_dir / "test-doc"
            doc_dir.mkdir()
            embedding = _make_fake_embedding(256)
            (doc_dir / "chunks.json").write_text('[{"chunk_index": 0, "text": "Test chunk"}]')
            (doc_dir / "embeddings.json").write_text(
                json.dumps([{"chunk_index": 0, "text": "Test chunk", "embedding": embedding}])
            )

            # Pre-populate memory with history
            store = FileSessionStore(storage_dir=memory_dir)
            store.add_turn("session-123", "user", "What is the dataset about?")
            store.add_turn("session-123", "assistant", "The dataset contains customer data.")

            # Run agent with session_id
            result = run_agent(
                question="Tell me more about the customers",
                doc_id="test-doc",
                embeddings_client=FakeEmbeddingsClient(),
                llm_client=FakeLLMClient(),
                storage_dir=storage_dir,
                session_id="session-123",
                memory_store=store,
            )

            # Check that conversation history was loaded
            assert len(result.get("conversation_history", [])) == 2
            assert result["conversation_history"][0]["role"] == "user"

            # Check trace event for memory load
            trace_types = [e["event_type"] for e in result.get("trace_events", [])]
            assert "MEMORY_LOADED" in trace_types

    def test_agent_saves_new_turns(self):
        """Agent should save user question and assistant response to memory."""
        import json

        from packages.agent.fake_clients import FakeEmbeddingsClient, FakeLLMClient
        from packages.agent.graph import run_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "contexts"
            storage_dir.mkdir()
            memory_dir = Path(tmpdir) / "sessions"
            memory_dir.mkdir()

            doc_dir = storage_dir / "test-doc"
            doc_dir.mkdir()
            embedding = _make_fake_embedding(256)
            (doc_dir / "chunks.json").write_text('[{"chunk_index": 0, "text": "Test"}]')
            (doc_dir / "embeddings.json").write_text(
                json.dumps([{"chunk_index": 0, "text": "Test", "embedding": embedding}])
            )

            store = FileSessionStore(storage_dir=memory_dir)

            run_agent(
                question="What is this?",
                doc_id="test-doc",
                embeddings_client=FakeEmbeddingsClient(),
                llm_client=FakeLLMClient(),
                storage_dir=storage_dir,
                session_id="new-session",
                memory_store=store,
            )

            # Check that turns were saved
            session = store.get_session("new-session")
            assert session is not None
            assert len(session.turns) == 2  # user + assistant
            assert session.turns[0].role == "user"
            assert session.turns[0].content == "What is this?"
            assert session.turns[1].role == "assistant"

    def test_agent_works_without_session_id(self):
        """Agent should work normally without session_id (no memory ops)."""
        import json

        from packages.agent.fake_clients import FakeEmbeddingsClient, FakeLLMClient
        from packages.agent.graph import run_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "contexts"
            storage_dir.mkdir()

            doc_dir = storage_dir / "test-doc"
            doc_dir.mkdir()
            embedding = _make_fake_embedding(256)
            (doc_dir / "chunks.json").write_text('[{"chunk_index": 0, "text": "Test"}]')
            (doc_dir / "embeddings.json").write_text(
                json.dumps([{"chunk_index": 0, "text": "Test", "embedding": embedding}])
            )

            result = run_agent(
                question="Hello",
                doc_id="test-doc",
                embeddings_client=FakeEmbeddingsClient(),
                llm_client=FakeLLMClient(),
                storage_dir=storage_dir,
                session_id=None,  # No session
            )

            # Should still work, just skip memory
            assert result.get("llm_response") is not None
            trace_types = [e["event_type"] for e in result.get("trace_events", [])]
            assert "MEMORY_SKIPPED" in trace_types

