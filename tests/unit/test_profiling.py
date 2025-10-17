"""Unit tests for botds.tools.profiling module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from botds.tools.profiling import SchemaProfiler, QualityGuard
from botds.utils import save_pickle, load_json


def load_profile_from_result(result):
    """Helper to load profile from profiler result."""
    return load_json(result["profile_ref"])


class TestSchemaProfiler:
    """Test SchemaProfiler functionality."""

    @pytest.fixture
    def profiler(self, tmp_path):
        """Create a SchemaProfiler instance."""
        return SchemaProfiler(tmp_path / "artifacts")

    @pytest.fixture
    def sample_df_ref(self, tmp_path):
        """Create a sample DataFrame and return its reference."""
        df = pd.DataFrame({
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "categorical": ["a", "b", "a", "c", "b"],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "test_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    @pytest.fixture
    def df_with_missing_ref(self, tmp_path):
        """Create a DataFrame with missing values."""
        df = pd.DataFrame({
            "col1": [1, 2, np.nan, 4, 5],
            "col2": ["a", None, "c", "d", "e"],
            "col3": [1.1, 2.2, 3.3, np.nan, 5.5]
        })

        df_path = tmp_path / "missing_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        artifacts_dir = tmp_path / "artifacts"
        profiler = SchemaProfiler(artifacts_dir)

        assert profiler.artifacts_dir == artifacts_dir
        assert profiler.handoffs_dir.exists()
        assert profiler.handoffs_dir == artifacts_dir / "handoffs"

    def test_profile_basic_structure(self, profiler, sample_df_ref):
        """Test basic profile structure."""
        result = profiler.profile(sample_df_ref)

        # Check result structure
        assert "profile_ref" in result
        assert "hash" in result
        assert "summary" in result

        # Load the actual profile from file
        from botds.utils import load_json
        profile = load_json(result["profile_ref"])

        # Check profile keys
        assert "shape" in profile
        assert "columns" in profile
        assert "dtypes" in profile
        assert "memory_usage_mb" in profile
        assert "missing_values" in profile
        assert "column_types" in profile

    def test_profile_shape(self, profiler, sample_df_ref):
        """Test shape information in profile."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        assert profile["shape"]["rows"] == 5
        assert profile["shape"]["columns"] == 4

    def test_profile_columns(self, profiler, sample_df_ref):
        """Test column information."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        assert profile["columns"] == ["numeric1", "numeric2", "categorical", "target"]
        assert "numeric1" in profile["dtypes"]
        assert "categorical" in profile["dtypes"]

    def test_profile_missing_values(self, profiler, df_with_missing_ref):
        """Test missing values analysis."""
        result = profiler.profile(df_with_missing_ref)
        profile = load_profile_from_result(result)

        missing = profile["missing_values"]
        assert "total_missing" in missing
        assert "missing_by_column" in missing
        assert "missing_percentage" in missing

        # Should detect missing values
        assert missing["total_missing"] > 0
        assert "col1" in missing["missing_by_column"] or "col3" in missing["missing_by_column"]

    def test_profile_no_missing_values(self, profiler, sample_df_ref):
        """Test profile when no missing values."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        missing = profile["missing_values"]
        assert missing["total_missing"] == 0
        assert len(missing["missing_by_column"]) == 0

    def test_profile_column_types(self, profiler, sample_df_ref):
        """Test column type classification."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        col_types = profile["column_types"]
        assert "numeric" in col_types
        assert "categorical" in col_types
        assert "datetime" in col_types

        # Check numeric columns
        assert "numeric1" in col_types["numeric"]
        assert "numeric2" in col_types["numeric"]
        assert "target" in col_types["numeric"]

        # Check categorical columns
        assert "categorical" in col_types["categorical"]

    def test_profile_numeric_stats(self, profiler, sample_df_ref):
        """Test numeric statistics."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        assert "numeric_stats" in profile
        numeric_stats = profile["numeric_stats"]

        # Should have stats for numeric columns
        assert "numeric1" in numeric_stats
        assert "numeric2" in numeric_stats

        # Check that standard stats are present
        assert "mean" in numeric_stats["numeric1"]
        assert "std" in numeric_stats["numeric1"]
        assert "min" in numeric_stats["numeric1"]
        assert "max" in numeric_stats["numeric1"]

    def test_profile_categorical_stats(self, profiler, sample_df_ref):
        """Test categorical statistics."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        assert "categorical_stats" in profile
        cat_stats = profile["categorical_stats"]

        # Should have stats for categorical column
        assert "categorical" in cat_stats

        cat_col_stats = cat_stats["categorical"]
        assert "unique_values" in cat_col_stats
        assert "most_frequent" in cat_col_stats
        assert "most_frequent_count" in cat_col_stats
        assert "top_5_values" in cat_col_stats

        # Check values
        assert cat_col_stats["unique_values"] == 3  # a, b, c

    def test_profile_memory_usage(self, profiler, sample_df_ref):
        """Test memory usage calculation."""
        result = profiler.profile(sample_df_ref)
        profile = load_profile_from_result(result)

        assert "memory_usage_mb" in profile
        assert isinstance(profile["memory_usage_mb"], (int, float))
        assert profile["memory_usage_mb"] > 0

    def test_profile_with_datetime(self, profiler, tmp_path):
        """Test profiling with datetime columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "value": [1, 2, 3, 4, 5]
        })

        df_path = tmp_path / "datetime_data.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        col_types = profile["column_types"]
        assert "date" in col_types["datetime"]
        assert "value" in col_types["numeric"]

    def test_profile_large_dataset(self, profiler, tmp_path):
        """Test profiling with larger dataset."""
        # Create a larger dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "col1": np.random.randn(1000),
            "col2": np.random.choice(["A", "B", "C"], 1000),
            "col3": np.random.randint(0, 100, 1000)
        })

        df_path = tmp_path / "large_data.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        assert profile["shape"]["rows"] == 1000
        assert profile["shape"]["columns"] == 3

    def test_profile_empty_dataframe(self, profiler, tmp_path):
        """Test profiling empty DataFrame."""
        df = pd.DataFrame()

        df_path = tmp_path / "empty_data.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        assert profile["shape"]["rows"] == 0
        assert profile["shape"]["columns"] == 0

    def test_profile_single_column(self, profiler, tmp_path):
        """Test profiling DataFrame with single column."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        df_path = tmp_path / "single_col.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        assert profile["shape"]["columns"] == 1
        assert profile["columns"] == ["col1"]

    def test_profile_high_cardinality_categorical(self, profiler, tmp_path):
        """Test profiling with high cardinality categorical."""
        # Create categorical with many unique values
        df = pd.DataFrame({
            "id": range(100),
            "category": [f"cat_{i}" for i in range(100)]
        })

        df_path = tmp_path / "high_card.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        # Should still profile successfully
        assert "categorical_stats" in profile or "numeric_stats" in profile

    def test_profile_mixed_types(self, profiler, tmp_path):
        """Test profiling with mixed data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": pd.date_range("2020-01-01", periods=3)
        })

        df_path = tmp_path / "mixed_types.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        assert profile["shape"]["columns"] == 5
        assert len(profile["column_types"]["numeric"]) >= 2  # int and float
        assert len(profile["column_types"]["datetime"]) == 1

    def test_profile_all_missing(self, profiler, tmp_path):
        """Test profiling column with all missing values."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [np.nan, np.nan, np.nan]
        })

        df_path = tmp_path / "all_missing.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        missing = profile["missing_values"]
        assert missing["total_missing"] == 3
        assert "col2" in missing["missing_by_column"]
        assert missing["missing_by_column"]["col2"] == 3


    def test_profile_potential_targets_classification(self, profiler, tmp_path):
        """Test identification of potential classification targets."""
        df = pd.DataFrame({
            "feature1": range(100),
            "binary_target": [0, 1] * 50,  # 2 unique values
            "multiclass_target": [0, 1, 2, 3, 4] * 20,  # 5 unique values
            "too_many_classes": range(100)  # 100 unique values
        })

        df_path = tmp_path / "targets.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        assert "potential_targets" in profile
        targets = profile["potential_targets"]

        # Should identify binary and multiclass as classification candidates
        target_cols = [t["column"] for t in targets]
        assert "binary_target" in target_cols
        assert "multiclass_target" in target_cols

        # Check types
        for target in targets:
            if target["column"] == "binary_target":
                assert target["type"] == "classification_candidate"
                assert target["unique_values"] == 2

    def test_profile_potential_targets_regression(self, profiler, tmp_path):
        """Test identification of potential regression targets."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": range(100),
            "continuous_target": np.random.randn(100) * 100,  # Many unique values
            "price": np.random.uniform(10, 1000, 100)
        })

        df_path = tmp_path / "regression_targets.pkl"
        save_pickle(df, df_path)

        result = profiler.profile(str(df_path))
        profile = load_profile_from_result(result)

        targets = profile["potential_targets"]
        target_cols = [t["column"] for t in targets]

        # Should identify continuous columns as regression candidates
        assert "continuous_target" in target_cols or "price" in target_cols

        # Check types
        for target in targets:
            if target["column"] in ["continuous_target", "price"]:
                assert target["type"] == "regression_candidate"
                assert target["unique_values"] > 20

    def test_profile_summary(self, profiler, sample_df_ref):
        """Test summary information in result."""
        result = profiler.profile(sample_df_ref)

        summary = result["summary"]
        assert "rows" in summary
        assert "columns" in summary
        assert "missing_total" in summary
        assert "numeric_columns" in summary
        assert "categorical_columns" in summary

        assert summary["rows"] == 5
        assert summary["columns"] == 4
        assert summary["missing_total"] == 0

    def test_get_function_definitions(self, profiler):
        """Test OpenAI function definitions."""
        defs = profiler.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) == 1

        func_def = defs[0]
        assert func_def["type"] == "function"
        assert "function" in func_def

        func = func_def["function"]
        assert func["name"] == "SchemaProfiler_profile"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert "df_ref" in params["properties"]
        assert "df_ref" in params["required"]



class TestQualityGuard:
    """Test QualityGuard functionality."""

    @pytest.fixture
    def quality_guard(self, tmp_path):
        """Create a QualityGuard instance."""
        return QualityGuard(tmp_path / "artifacts")

    def test_init(self, tmp_path):
        """Test QualityGuard initialization."""
        artifacts_dir = tmp_path / "artifacts"
        guard = QualityGuard(artifacts_dir)

        assert guard.artifacts_dir == artifacts_dir

    def test_leakage_scan_pass(self, quality_guard, tmp_path):
        """Test leakage scan with clean data."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "clean_data.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="iid"
        )

        assert "status" in result
        assert "offenders" in result
        assert "warnings" in result
        assert "summary" in result

        # Clean data should pass (though may have warnings for suspicious names)
        assert result["status"] in ["pass", "warn"]
        assert len(result["offenders"]) == 0

    def test_leakage_scan_perfect_correlation(self, quality_guard, tmp_path):
        """Test detection of perfect correlation."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "leaky_feature": [10, 20, 30, 40, 50],  # Perfect correlation with target
            "target": [10, 20, 30, 40, 50]
        })

        df_path = tmp_path / "leaky_data.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="iid"
        )

        assert result["status"] == "block"
        assert len(result["offenders"]) > 0

        # Check offender details - both features have perfect correlation
        offender_cols = [o["column"] for o in result["offenders"]]
        assert "leaky_feature" in offender_cols or "feature1" in offender_cols

        for offender in result["offenders"]:
            assert offender["issue"] == "perfect_correlation"
            assert abs(offender["correlation"]) > 0.99
            assert offender["severity"] == "block"

    def test_leakage_scan_high_correlation(self, quality_guard, tmp_path):
        """Test detection of high correlation."""
        np.random.seed(42)
        target = np.array([1, 2, 3, 4, 5], dtype=float)
        high_corr = target + np.random.randn(5) * 0.1  # High but not perfect correlation

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "high_corr_feature": high_corr,
            "target": target
        })

        df_path = tmp_path / "high_corr_data.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="iid"
        )

        # Should have warnings for high correlation
        assert result["status"] in ["warn", "block"]
        if result["status"] == "warn":
            assert len(result["warnings"]) > 0

    def test_leakage_scan_missing_target(self, quality_guard, tmp_path):
        """Test leakage scan with missing target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10]
        })

        df_path = tmp_path / "no_target.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="nonexistent_target",
            split_policy="iid"
        )

        assert result["status"] == "block"
        assert len(result["offenders"]) == 0
        assert "reason" in result
        assert "not found" in result["reason"]

    def test_leakage_scan_suspicious_names(self, quality_guard, tmp_path):
        """Test detection of suspicious column names."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "created_timestamp": [100, 200, 300, 400, 500],
            "prediction_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "suspicious_names.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="iid"
        )

        # Should have warnings for suspicious names
        assert len(result["warnings"]) > 0

        # Check for specific patterns
        warning_cols = [w["column"] for w in result["warnings"]]
        assert any("id" in col.lower() or "timestamp" in col.lower() or "prediction" in col.lower()
                   for col in warning_cols)

    def test_leakage_scan_time_policy_valid(self, quality_guard, tmp_path):
        """Test leakage scan with time-based split policy."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "timestamp": pd.date_range("2020-01-01", periods=5),
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "time_data.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="time",
            time_col="timestamp"
        )

        assert "status" in result
        # Should not have blocking issues for valid time column
        blocking_issues = [o for o in result["offenders"] if o["issue"] == "missing_time_column"]
        assert len(blocking_issues) == 0

    def test_leakage_scan_time_policy_missing_column(self, quality_guard, tmp_path):
        """Test leakage scan with missing time column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "no_time.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="time",
            time_col="timestamp"
        )

        assert result["status"] == "block"
        assert len(result["offenders"]) > 0

        # Check for missing time column error
        offender = result["offenders"][0]
        assert offender["issue"] == "missing_time_column"
        assert offender["severity"] == "block"

    def test_leakage_scan_time_policy_invalid_format(self, quality_guard, tmp_path):
        """Test leakage scan with invalid datetime format."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "timestamp": ["not", "a", "valid", "date", "format"],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "invalid_time.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="time",
            time_col="timestamp"
        )

        # Should have warnings for invalid datetime format
        assert len(result["warnings"]) > 0 or len(result["offenders"]) > 0

    def test_leakage_scan_summary(self, quality_guard, tmp_path):
        """Test summary information in leakage scan result."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        })

        df_path = tmp_path / "data.pkl"
        save_pickle(df, df_path)

        result = quality_guard.leakage_scan(
            df_ref=str(df_path),
            target="target",
            split_policy="iid"
        )

        summary = result["summary"]
        assert "blocking_issues" in summary
        assert "warnings" in summary
        assert "columns_checked" in summary

        # Should check all columns except target
        assert summary["columns_checked"] == 2

    def test_get_function_definitions(self, quality_guard):
        """Test OpenAI function definitions."""
        defs = quality_guard.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) == 1

        func_def = defs[0]
        assert func_def["type"] == "function"
        assert "function" in func_def

        func = func_def["function"]
        assert func["name"] == "QualityGuard_leakage_scan"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert "df_ref" in params["properties"]
        assert "target" in params["properties"]
        assert "split_policy" in params["properties"]
        assert "time_col" in params["properties"]

        # Check required fields
        assert "df_ref" in params["required"]
        assert "target" in params["required"]
        assert "split_policy" in params["required"]

