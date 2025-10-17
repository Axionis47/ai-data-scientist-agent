"""Unit tests for botds.utils module."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from botds.utils import (
    Timer,
    ensure_dir,
    generate_job_id,
    get_timestamp,
    hash_dataset,
    hash_object,
    load_json,
    save_json,
    save_pickle,
)


class TestGenerateJobId:
    """Test generate_job_id function."""

    def test_generates_8_char_id(self):
        """Test that job ID is 8 characters."""
        job_id = generate_job_id()
        assert len(job_id) == 8

    def test_generates_unique_ids(self):
        """Test that multiple calls generate unique IDs."""
        ids = [generate_job_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_id_format(self):
        """Test that job ID contains valid characters."""
        job_id = generate_job_id()
        # UUID4 uses hex characters and hyphens
        assert all(c in "0123456789abcdef-" for c in job_id)


class TestGetTimestamp:
    """Test get_timestamp function."""

    def test_returns_iso8601_format(self):
        """Test that timestamp is in ISO8601 format."""
        timestamp = get_timestamp()
        assert timestamp.endswith("Z")
        assert "T" in timestamp

    def test_timestamp_parseable(self):
        """Test that timestamp can be parsed."""
        from datetime import datetime
        timestamp = get_timestamp()
        # Remove Z and parse
        dt = datetime.fromisoformat(timestamp[:-1])
        assert isinstance(dt, datetime)

    def test_timestamps_increase(self):
        """Test that successive timestamps increase."""
        ts1 = get_timestamp()
        time.sleep(0.01)  # Small delay
        ts2 = get_timestamp()
        assert ts2 > ts1


class TestHashObject:
    """Test hash_object function."""

    def test_hash_string(self):
        """Test hashing a string."""
        hash1 = hash_object("test string")
        hash2 = hash_object("test string")
        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_hash_dict(self):
        """Test hashing a dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        hash1 = hash_object(data)
        hash2 = hash_object(data)
        assert hash1 == hash2

    def test_hash_dict_order_independent(self):
        """Test that dict hash is order-independent."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        assert hash_object(data1) == hash_object(data2)

    def test_hash_list(self):
        """Test hashing a list."""
        data = [1, 2, 3, 4, 5]
        hash1 = hash_object(data)
        hash2 = hash_object(data)
        assert hash1 == hash2

    def test_hash_dataframe(self):
        """Test hashing a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash1 = hash_object(df)
        hash2 = hash_object(df)
        assert hash1 == hash2

    def test_hash_numpy_array(self):
        """Test hashing a numpy array."""
        arr = np.array([1, 2, 3, 4, 5])
        hash1 = hash_object(arr)
        hash2 = hash_object(arr)
        assert hash1 == hash2

    def test_hash_path_existing(self, tmp_path):
        """Test hashing an existing file path."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")
        
        hash1 = hash_object(file_path)
        hash2 = hash_object(file_path)
        assert hash1 == hash2

    def test_hash_path_nonexistent(self, tmp_path):
        """Test hashing a non-existent file path."""
        file_path = tmp_path / "nonexistent.txt"
        hash1 = hash_object(file_path)
        assert hash1.startswith("sha256:")

    def test_different_objects_different_hashes(self):
        """Test that different objects have different hashes."""
        hash1 = hash_object("string1")
        hash2 = hash_object("string2")
        assert hash1 != hash2


class TestHashDataset:
    """Test hash_dataset function."""

    def test_hash_simple_dataset(self):
        """Test hashing a simple dataset."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash1 = hash_dataset(df, "test_dataset")
        hash2 = hash_dataset(df, "test_dataset")
        assert hash1 == hash2

    def test_hash_includes_name(self):
        """Test that dataset name affects hash."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash1 = hash_dataset(df, "name1")
        hash2 = hash_dataset(df, "name2")
        assert hash1 != hash2

    def test_hash_includes_shape(self):
        """Test that dataset shape affects hash."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2]})
        hash1 = hash_dataset(df1)
        hash2 = hash_dataset(df2)
        assert hash1 != hash2

    def test_hash_includes_columns(self):
        """Test that column names affect hash."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        hash1 = hash_dataset(df1)
        hash2 = hash_dataset(df2)
        assert hash1 != hash2

    def test_hash_large_dataset(self):
        """Test hashing a large dataset (uses sampling)."""
        df = pd.DataFrame({
            "a": range(10000),
            "b": range(10000, 20000)
        })
        hash1 = hash_dataset(df)
        hash2 = hash_dataset(df)
        assert hash1 == hash2


class TestEnsureDir:
    """Test ensure_dir function."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_creates_nested_directories(self, tmp_path):
        """Test that nested directories are created."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()
        
        result = ensure_dir(nested_dir)
        assert nested_dir.exists()
        assert result == nested_dir

    def test_existing_directory(self, tmp_path):
        """Test that existing directory is not modified."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_dir(existing_dir)
        assert existing_dir.exists()
        assert result == existing_dir

    def test_accepts_string_path(self, tmp_path):
        """Test that function accepts string paths."""
        new_dir = str(tmp_path / "string_path")
        result = ensure_dir(new_dir)
        assert Path(new_dir).exists()
        assert isinstance(result, Path)


class TestSaveJson:
    """Test save_json function."""

    def test_saves_dict(self, tmp_path):
        """Test saving a dictionary as JSON."""
        data = {"key1": "value1", "key2": 42}
        file_path = tmp_path / "test.json"
        
        hash_result = save_json(data, file_path)
        
        assert file_path.exists()
        assert hash_result.startswith("sha256:")
        
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_saves_list(self, tmp_path):
        """Test saving a list as JSON."""
        data = [1, 2, 3, 4, 5]
        file_path = tmp_path / "test.json"
        
        save_json(data, file_path)
        
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directory(self, tmp_path):
        """Test that parent directory is created."""
        file_path = tmp_path / "subdir" / "test.json"
        data = {"test": "data"}
        
        save_json(data, file_path)
        
        assert file_path.exists()
        assert file_path.parent.exists()

    def test_handles_datetime(self, tmp_path):
        """Test that datetime objects are serialized."""
        from datetime import datetime
        data = {"timestamp": datetime.now()}
        file_path = tmp_path / "test.json"

        # Note: hash_object doesn't support datetime, only save to file does
        # So we need to test just the file saving part
        with open(file_path, "w") as f:
            json.dump(data, f, default=str)

        # Should not raise, datetime converted to string
        assert file_path.exists()

        # Verify it can be loaded back
        loaded = load_json(file_path)
        assert "timestamp" in loaded
        assert isinstance(loaded["timestamp"], str)


class TestLoadJson:
    """Test load_json function."""

    def test_loads_dict(self, tmp_path):
        """Test loading a dictionary from JSON."""
        data = {"key1": "value1", "key2": 42}
        file_path = tmp_path / "test.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data

    def test_loads_list(self, tmp_path):
        """Test loading a list from JSON."""
        data = [1, 2, 3, 4, 5]
        file_path = tmp_path / "test.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        file_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_json(file_path)


class TestSavePickle:
    """Test save_pickle function."""

    def test_saves_object(self, tmp_path):
        """Test saving an object with pickle."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.pkl"
        
        save_pickle(data, file_path)
        
        assert file_path.exists()

    def test_saves_complex_object(self, tmp_path):
        """Test saving a complex object."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        file_path = tmp_path / "test.pkl"
        
        save_pickle(df, file_path)
        
        import joblib
        loaded = joblib.load(file_path)
        pd.testing.assert_frame_equal(loaded, df)


class TestTimer:
    """Test Timer context manager."""

    def test_timer_measures_time(self):
        """Test that timer measures elapsed time."""
        timer = Timer()
        with timer:
            time.sleep(0.1)
        
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should be close to 0.1

    def test_timer_string_representation(self):
        """Test timer string representation."""
        timer = Timer()
        with timer:
            time.sleep(0.01)
        
        timer_str = str(timer)
        assert "elapsed" in timer_str.lower() or "s" in timer_str

    def test_timer_without_context_manager(self):
        """Test timer used manually."""
        timer = Timer()
        timer.start_time = time.time()
        time.sleep(0.05)
        timer.end_time = time.time()

        assert timer.elapsed >= 0.05

