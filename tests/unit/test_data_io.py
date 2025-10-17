"""Unit tests for botds.tools.data_io module."""

import pytest
import pandas as pd
from pathlib import Path

from botds.tools.data_io import DataStore


class TestDataStore:
    """Test DataStore functionality."""

    @pytest.fixture
    def data_store(self, tmp_path):
        """Create a DataStore instance."""
        return DataStore(tmp_path / "artifacts")

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        artifacts_dir = tmp_path / "artifacts"
        store = DataStore(artifacts_dir)
        
        assert store.artifacts_dir == artifacts_dir
        assert store.data_dir.exists()
        assert store.data_dir == artifacts_dir / "data"

    def test_read_builtin_iris(self, data_store):
        """Test loading iris dataset."""
        result = data_store.read_builtin("iris")
        
        assert "df_ref" in result
        assert "target" in result
        assert "task_hint" in result
        assert "shape" in result
        assert "hash" in result
        
        assert result["target"] == "target"
        assert result["task_hint"] == "classification"
        assert result["shape"] == (150, 5)  # 4 features + 1 target
        
        # Verify file was created
        df_path = Path(result["df_ref"])
        assert df_path.exists()
        
        # Verify we can load it back
        df = data_store.load_dataframe(result["df_ref"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert "target" in df.columns

    def test_read_builtin_breast_cancer(self, data_store):
        """Test loading breast cancer dataset."""
        result = data_store.read_builtin("breast_cancer")
        
        assert result["target"] == "target"
        assert result["task_hint"] == "classification"
        assert result["shape"] == (569, 31)  # 30 features + 1 target
        
        df = data_store.load_dataframe(result["df_ref"])
        assert len(df) == 569

    def test_read_builtin_diabetes(self, data_store):
        """Test loading diabetes dataset."""
        result = data_store.read_builtin("diabetes")
        
        assert result["target"] == "target"
        assert result["task_hint"] == "regression"
        assert result["shape"] == (442, 11)  # 10 features + 1 target
        
        df = data_store.load_dataframe(result["df_ref"])
        assert len(df) == 442

    def test_read_builtin_unknown_dataset(self, data_store):
        """Test loading unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown builtin dataset"):
            data_store.read_builtin("unknown_dataset")

    def test_read_csv_single_file(self, data_store, tmp_path):
        """Test loading a single CSV file."""
        # Create a test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        df.to_csv(csv_path, index=False)
        
        result = data_store.read_csv([str(csv_path)])
        
        assert "df_ref" in result
        assert "shape" in result
        assert "hash" in result
        assert result["shape"] == (3, 2)
        
        # Verify we can load it back
        loaded_df = data_store.load_dataframe(result["df_ref"])
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["col1", "col2"]

    def test_read_csv_multiple_files(self, data_store, tmp_path):
        """Test loading and combining multiple CSV files."""
        # Create two test CSVs
        csv1_path = tmp_path / "test1.csv"
        df1 = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"]
        })
        df1.to_csv(csv1_path, index=False)
        
        csv2_path = tmp_path / "test2.csv"
        df2 = pd.DataFrame({
            "col1": [3, 4],
            "col2": ["c", "d"]
        })
        df2.to_csv(csv2_path, index=False)
        
        result = data_store.read_csv([str(csv1_path), str(csv2_path)])
        
        assert result["shape"] == (4, 2)  # Combined rows
        
        # Verify combined data
        loaded_df = data_store.load_dataframe(result["df_ref"])
        assert len(loaded_df) == 4
        assert loaded_df["col1"].tolist() == [1, 2, 3, 4]

    def test_read_csv_empty_paths(self, data_store):
        """Test that empty paths list raises error."""
        with pytest.raises(ValueError, match="No CSV paths provided"):
            data_store.read_csv([])

    def test_read_csv_nonexistent_file(self, data_store):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            data_store.read_csv(["/nonexistent/file.csv"])

    def test_load_dataframe(self, data_store):
        """Test loading DataFrame from reference."""
        # First create a dataset
        result = data_store.read_builtin("iris")
        
        # Load it back
        df = data_store.load_dataframe(result["df_ref"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert "target" in df.columns

    def test_get_function_definitions(self, data_store):
        """Test getting OpenAI function definitions."""
        functions = data_store.get_function_definitions()
        
        assert isinstance(functions, list)
        assert len(functions) == 2
        
        # Check read_builtin function
        builtin_func = functions[0]
        assert builtin_func["type"] == "function"
        assert builtin_func["function"]["name"] == "DataStore_read_builtin"
        assert "iris" in builtin_func["function"]["parameters"]["properties"]["name"]["enum"]
        
        # Check read_csv function
        csv_func = functions[1]
        assert csv_func["type"] == "function"
        assert csv_func["function"]["name"] == "DataStore_read_csv"
        assert "paths" in csv_func["function"]["parameters"]["properties"]

    def test_hash_consistency(self, data_store):
        """Test that same dataset produces same hash."""
        result1 = data_store.read_builtin("iris")
        result2 = data_store.read_builtin("iris")
        
        # Hashes should be the same for the same dataset
        assert result1["hash"] == result2["hash"]

    def test_different_datasets_different_hashes(self, data_store):
        """Test that different datasets produce different hashes."""
        result1 = data_store.read_builtin("iris")
        result2 = data_store.read_builtin("diabetes")
        
        assert result1["hash"] != result2["hash"]

