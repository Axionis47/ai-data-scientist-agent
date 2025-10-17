"""Unit tests for botds.context module."""

import json
from pathlib import Path

import pytest

from botds.context import DataCard, DecisionLog, HandoffLedger, RunManifest


class TestDecisionLog:
    """Test DecisionLog functionality."""

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that initialization creates parent directory."""
        log_path = tmp_path / "logs" / "decision_log.jsonl"
        log = DecisionLog(log_path)
        
        assert log_path.parent.exists()
        assert log.log_path == log_path

    def test_record_decision(self, tmp_path):
        """Test recording a decision."""
        log_path = tmp_path / "decision_log.jsonl"
        log = DecisionLog(log_path)
        
        log.record_decision(
            stage="model_selection",
            decision="Use RandomForest",
            rationale="Best performance on validation set",
            inputs_refs=["profile.json", "split_indices.json"],
            auth_model="openai"
        )
        
        assert log_path.exists()
        
        with open(log_path) as f:
            line = f.readline()
            entry = json.loads(line)
        
        assert entry["stage"] == "model_selection"
        assert entry["decision"] == "Use RandomForest"
        assert entry["rationale"] == "Best performance on validation set"
        assert entry["auth_model"] == "openai"
        assert "timestamp" in entry

    def test_record_multiple_decisions(self, tmp_path):
        """Test recording multiple decisions."""
        log_path = tmp_path / "decision_log.jsonl"
        log = DecisionLog(log_path)
        
        log.record_decision("stage1", "decision1", "rationale1", [])
        log.record_decision("stage2", "decision2", "rationale2", [])
        log.record_decision("stage3", "decision3", "rationale3", [])
        
        with open(log_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 3

    def test_get_decisions_empty(self, tmp_path):
        """Test getting decisions from empty log."""
        log_path = tmp_path / "decision_log.jsonl"
        log = DecisionLog(log_path)
        
        decisions = log.get_decisions()
        assert decisions == []

    def test_get_decisions(self, tmp_path):
        """Test getting all decisions."""
        log_path = tmp_path / "decision_log.jsonl"
        log = DecisionLog(log_path)
        
        log.record_decision("stage1", "decision1", "rationale1", ["ref1"])
        log.record_decision("stage2", "decision2", "rationale2", ["ref2"])
        
        decisions = log.get_decisions()
        
        assert len(decisions) == 2
        assert decisions[0]["stage"] == "stage1"
        assert decisions[1]["stage"] == "stage2"

    def test_decision_log_persistence(self, tmp_path):
        """Test that decisions persist across instances."""
        log_path = tmp_path / "decision_log.jsonl"
        
        # First instance
        log1 = DecisionLog(log_path)
        log1.record_decision("stage1", "decision1", "rationale1", [])
        
        # Second instance
        log2 = DecisionLog(log_path)
        decisions = log2.get_decisions()
        
        assert len(decisions) == 1
        assert decisions[0]["stage"] == "stage1"


class TestDataCard:
    """Test DataCard functionality."""

    def test_init(self):
        """Test DataCard initialization."""
        card = DataCard(name="iris", version="v1")
        
        assert card.name == "iris"
        assert card.version == "v1"
        assert card.created_at is not None
        assert card.metadata == {}
        assert card.quality_notes == []
        assert card.leakage_rules == []

    def test_add_metadata(self):
        """Test adding metadata."""
        card = DataCard(name="test")
        
        card.add_metadata("n_rows", 150)
        card.add_metadata("n_cols", 5)
        
        assert card.metadata["n_rows"] == 150
        assert card.metadata["n_cols"] == 5

    def test_add_quality_note(self):
        """Test adding quality notes."""
        card = DataCard(name="test")
        
        card.add_quality_note("No missing values")
        card.add_quality_note("Balanced classes")
        
        assert len(card.quality_notes) == 2
        assert "No missing values" in card.quality_notes

    def test_add_leakage_rule(self):
        """Test adding leakage rules."""
        card = DataCard(name="test")
        
        card.add_leakage_rule("Drop 'id' column")
        card.add_leakage_rule("Drop 'timestamp' column")
        
        assert len(card.leakage_rules) == 2
        assert "Drop 'id' column" in card.leakage_rules

    def test_to_dict(self):
        """Test converting to dictionary."""
        card = DataCard(name="test", version="v2")
        card.add_metadata("key", "value")
        card.add_quality_note("note1")
        card.add_leakage_rule("rule1")
        
        data = card.to_dict()
        
        assert data["name"] == "test"
        assert data["version"] == "v2"
        assert "created_at" in data
        assert data["metadata"]["key"] == "value"
        assert "note1" in data["quality_notes"]
        assert "rule1" in data["leakage_rules"]

    def test_save(self, tmp_path):
        """Test saving data card."""
        card = DataCard(name="test")
        card.add_metadata("test_key", "test_value")
        
        save_path = tmp_path / "data_card.json"
        hash_result = card.save(save_path)
        
        assert save_path.exists()
        assert hash_result.startswith("sha256:")
        
        with open(save_path) as f:
            loaded = json.load(f)
        
        assert loaded["name"] == "test"
        assert loaded["metadata"]["test_key"] == "test_value"


class TestHandoffLedger:
    """Test HandoffLedger functionality."""

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that initialization creates parent directory."""
        ledger_path = tmp_path / "logs" / "handoff_ledger.jsonl"
        ledger = HandoffLedger(ledger_path)
        
        assert ledger_path.parent.exists()
        assert ledger.ledger_path == ledger_path

    def test_append_handoff(self, tmp_path):
        """Test appending a handoff entry."""
        ledger_path = tmp_path / "handoff_ledger.jsonl"
        ledger = HandoffLedger(ledger_path)

        ledger.append(
            job_id="test1234",
            stage="profiling",
            input_refs=["dataset.csv"],
            output_refs=["profile.json"],
            schema_uri="profile_v1",
            hash_value="sha256:def456"
        )

        assert ledger_path.exists()

        with open(ledger_path) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["job_id"] == "test1234"
        assert entry["stage"] == "profiling"
        assert "dataset.csv" in entry["inputs"]
        assert "profile.json" in entry["outputs"]
        assert entry["schema"] == "profile_v1"
        assert "timestamp" in entry

    def test_append_multiple_handoffs(self, tmp_path):
        """Test appending multiple handoffs."""
        ledger_path = tmp_path / "handoff_ledger.jsonl"
        ledger = HandoffLedger(ledger_path)

        ledger.append("job1", "stage1", [], [], "schema1", "hash1")
        ledger.append("job1", "stage2", [], [], "schema2", "hash2")
        ledger.append("job1", "stage3", [], [], "schema3", "hash3")

        with open(ledger_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_get_entries_empty(self, tmp_path):
        """Test getting entries from empty ledger."""
        ledger_path = tmp_path / "handoff_ledger.jsonl"
        ledger = HandoffLedger(ledger_path)

        entries = ledger.get_entries()
        assert entries == []

    def test_get_entries(self, tmp_path):
        """Test getting all entries."""
        ledger_path = tmp_path / "handoff_ledger.jsonl"
        ledger = HandoffLedger(ledger_path)

        ledger.append("job1", "stage1", ["in1"], ["out1"], "schema1", "hash1")
        ledger.append("job1", "stage2", ["in2"], ["out2"], "schema2", "hash2")

        entries = ledger.get_entries()

        assert len(entries) == 2
        assert entries[0]["stage"] == "stage1"
        assert entries[1]["stage"] == "stage2"


class TestRunManifest:
    """Test RunManifest functionality."""

    def test_init(self):
        """Test RunManifest initialization."""
        manifest = RunManifest(job_id="test1234")

        assert manifest.job_id == "test1234"
        assert manifest.created_at is not None
        assert manifest.seeds == {}
        assert manifest.shortcuts_taken == []
        assert manifest.config_hash is None
        assert manifest.dataset_hash is None

    def test_add_seed(self):
        """Test adding a seed."""
        manifest = RunManifest(job_id="test1234")

        manifest.add_seed("split", 42)
        manifest.add_seed("sampling", 123)

        assert manifest.seeds["split"] == 42
        assert manifest.seeds["sampling"] == 123

    def test_add_shortcut(self):
        """Test adding a shortcut."""
        manifest = RunManifest(job_id="test1234")

        manifest.add_shortcut("Skipped hyperparameter tuning due to time budget")

        assert len(manifest.shortcuts_taken) == 1
        assert "hyperparameter tuning" in manifest.shortcuts_taken[0]

    def test_set_hashes(self):
        """Test setting config and dataset hashes."""
        manifest = RunManifest(job_id="test1234")

        manifest.set_config_hash("sha256:config123")
        manifest.set_dataset_hash("sha256:dataset456")

        assert manifest.config_hash == "sha256:config123"
        assert manifest.dataset_hash == "sha256:dataset456"

    def test_to_dict(self):
        """Test converting to dictionary."""
        manifest = RunManifest(job_id="test1234")
        manifest.add_seed("split", 42)
        manifest.add_shortcut("shortcut1")
        manifest.set_config_hash("sha256:abc")

        data = manifest.to_dict()

        assert data["job_id"] == "test1234"
        assert "created_at" in data
        assert data["seeds"]["split"] == 42
        assert "shortcut1" in data["shortcuts_taken"]
        assert data["config_hash"] == "sha256:abc"

    def test_save(self, tmp_path):
        """Test saving manifest."""
        manifest = RunManifest(job_id="test1234")
        manifest.add_seed("split", 42)

        save_path = tmp_path / "run_manifest.json"
        hash_result = manifest.save(save_path)

        assert save_path.exists()
        assert hash_result.startswith("sha256:")

        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded["job_id"] == "test1234"
        assert loaded["seeds"]["split"] == 42

    def test_manifest_completeness(self):
        """Test that manifest contains all required fields."""
        manifest = RunManifest(job_id="test1234")
        manifest.add_seed("split", 42)
        manifest.add_seed("sampling", 123)
        manifest.add_shortcut("Skipped tuning")
        manifest.set_config_hash("sha256:config")
        manifest.set_dataset_hash("sha256:dataset")

        data = manifest.to_dict()

        # Required fields
        assert "job_id" in data
        assert "created_at" in data
        assert "seeds" in data
        assert "shortcuts_taken" in data
        assert "config_hash" in data
        assert "dataset_hash" in data

        # Values
        assert len(data["seeds"]) == 2
        assert len(data["shortcuts_taken"]) == 1

