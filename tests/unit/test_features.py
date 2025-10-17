"""Unit tests for botds.tools.features module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from botds.tools.features import Splitter, Featurizer
from botds.utils import save_pickle, load_pickle, load_json


def load_splits_from_result(result):
    """Helper to load split info from splitter result."""
    return load_json(result["splits_ref"])


class TestSplitter:
    """Test Splitter functionality."""

    @pytest.fixture
    def splitter(self, tmp_path):
        """Create a Splitter instance."""
        return Splitter(tmp_path / "artifacts")

    @pytest.fixture
    def classification_df_ref(self, tmp_path):
        """Create a classification dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })

        df_path = tmp_path / "classification_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    @pytest.fixture
    def regression_df_ref(self, tmp_path):
        """Create a regression dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randn(100)
        })

        df_path = tmp_path / "regression_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        artifacts_dir = tmp_path / "artifacts"
        splitter = Splitter(artifacts_dir)

        assert splitter.artifacts_dir == artifacts_dir
        assert splitter.handoffs_dir.exists()
        assert splitter.handoffs_dir == artifacts_dir / "handoffs"

    def test_make_splits_iid_basic(self, splitter, classification_df_ref):
        """Test basic IID splitting."""
        result = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Check result structure
        assert "splits_ref" in result
        assert "hash" in result
        assert "summary" in result

        # Load actual splits
        splits = load_splits_from_result(result)

        assert "indices" in splits
        assert "train" in splits["indices"]
        assert "val" in splits["indices"]
        assert "test" in splits["indices"]

        # Check that indices don't overlap
        train_idx = set(splits["indices"]["train"])
        val_idx = set(splits["indices"]["val"])
        test_idx = set(splits["indices"]["test"])

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

        # Check total coverage
        assert len(train_idx) + len(val_idx) + len(test_idx) == 100

    def test_make_splits_iid_sizes(self, splitter, classification_df_ref):
        """Test that split sizes are approximately correct."""
        result = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Check summary
        assert "summary" in result
        sizes = result["summary"]
        assert "train" in sizes
        assert "val" in sizes
        assert "test" in sizes

        # Load actual splits
        splits = load_splits_from_result(result)
        actual_sizes = splits["sizes"]

        # Test set should be 20% of 100 = 20 samples
        assert actual_sizes["test"] == 20

        # Val set should be 20% of remaining 80 = 16 samples (due to stratification, might be 16-20)
        assert actual_sizes["val"] >= 16 and actual_sizes["val"] <= 20

        # Train set should be the rest (60-64 samples)
        assert actual_sizes["train"] >= 60 and actual_sizes["train"] <= 64

    def test_make_splits_time_policy(self, splitter, regression_df_ref):
        """Test time-based splitting."""
        result = splitter.make_splits(
            df_ref=regression_df_ref,
            target="target",
            policy="time",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        train_idx = splits["indices"]["train"]
        val_idx = splits["indices"]["val"]
        test_idx = splits["indices"]["test"]

        # Time splits should be sequential
        assert max(train_idx) < min(val_idx)
        assert max(val_idx) < min(test_idx)

        # Check sizes
        assert len(test_idx) == 20  # Last 20%
        assert len(val_idx) == 16   # 20% of remaining 80
        assert len(train_idx) == 64  # First 64%

    def test_make_splits_custom_sizes(self, splitter, classification_df_ref):
        """Test custom split sizes."""
        result = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            test_size=0.3,
            val_size=0.1,
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)
        sizes = splits["sizes"]

        # Test set should be 30% of 100 = 30
        assert sizes["test"] == 30

        # Val set should be 10% of remaining 70 = 7 (due to stratification, might be 7-11)
        assert sizes["val"] >= 7 and sizes["val"] <= 11

        # Train set should be the rest (59-63)
        assert sizes["train"] >= 59 and sizes["train"] <= 63

    def test_make_splits_seed_reproducibility(self, splitter, classification_df_ref):
        """Test that same seed produces same splits."""
        result1 = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            seed=42
        )

        result2 = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            seed=42
        )

        # Load actual splits
        splits1 = load_splits_from_result(result1)
        splits2 = load_splits_from_result(result2)

        assert splits1["indices"]["train"] == splits2["indices"]["train"]
        assert splits1["indices"]["val"] == splits2["indices"]["val"]
        assert splits1["indices"]["test"] == splits2["indices"]["test"]

    def test_make_splits_different_seeds(self, splitter, classification_df_ref):
        """Test that different seeds produce different splits."""
        result1 = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            seed=42
        )

        # Load splits immediately (before they get overwritten)
        splits1 = load_splits_from_result(result1)
        train1 = splits1["indices"]["train"].copy()

        result2 = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            seed=123
        )

        # Load second splits
        splits2 = load_splits_from_result(result2)
        train2 = splits2["indices"]["train"]

        # Splits should be different
        assert train1 != train2

    def test_make_splits_missing_target(self, splitter, classification_df_ref):
        """Test error when target column doesn't exist."""
        with pytest.raises(ValueError, match="Target column .* not found"):
            splitter.make_splits(
                df_ref=classification_df_ref,
                target="nonexistent_target",
                policy="iid"
            )

    def test_make_splits_small_dataset(self, splitter, tmp_path):
        """Test splitting with small dataset."""
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        df_path = tmp_path / "small_data.pkl"
        save_pickle(df, df_path)

        result = splitter.make_splits(
            df_ref=str(df_path),
            target="target",
            policy="iid",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        # Should still work with small dataset
        assert "indices" in splits
        assert len(splits["indices"]["train"]) >= 1

    def test_make_splits_regression_no_stratify(self, splitter, regression_df_ref):
        """Test that regression doesn't use stratification."""
        # This should work without errors (regression targets can't be stratified)
        result = splitter.make_splits(
            df_ref=regression_df_ref,
            target="target",
            policy="iid",
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        assert "indices" in splits
        assert len(splits["indices"]["train"]) > 0

    def test_make_splits_multiclass(self, splitter, tmp_path):
        """Test splitting with multiclass target."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.choice([0, 1, 2, 3], 100)
        })

        df_path = tmp_path / "multiclass_data.pkl"
        save_pickle(df, df_path)

        result = splitter.make_splits(
            df_ref=str(df_path),
            target="target",
            policy="iid",
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        assert "indices" in splits
        assert len(splits["indices"]["train"]) > 0

    def test_make_splits_high_cardinality_target(self, splitter, tmp_path):
        """Test splitting with high cardinality target (no stratification)."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feature": np.random.randn(100),
            "target": range(100)  # 100 unique values
        })

        df_path = tmp_path / "high_card_target.pkl"
        save_pickle(df, df_path)

        result = splitter.make_splits(
            df_ref=str(df_path),
            target="target",
            policy="iid",
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        # Should work without stratification
        assert "indices" in splits

    def test_make_splits_time_sequential_order(self, splitter, tmp_path):
        """Test that time splits maintain sequential order."""
        df = pd.DataFrame({
            "time": pd.date_range("2020-01-01", periods=100),
            "value": range(100),
            "target": range(100)
        })

        df_path = tmp_path / "time_series.pkl"
        save_pickle(df, df_path)

        result = splitter.make_splits(
            df_ref=str(df_path),
            target="target",
            policy="time",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        train_idx = splits["indices"]["train"]
        val_idx = splits["indices"]["val"]
        test_idx = splits["indices"]["test"]

        # Verify sequential order
        assert train_idx == list(range(0, 64))
        assert val_idx == list(range(64, 80))
        assert test_idx == list(range(80, 100))

    def test_make_splits_metadata(self, splitter, classification_df_ref):
        """Test that split metadata is included."""
        result = splitter.make_splits(
            df_ref=classification_df_ref,
            target="target",
            policy="iid",
            test_size=0.2,
            val_size=0.2,
            seed=42
        )

        # Load actual splits
        splits = load_splits_from_result(result)

        assert splits["seed"] == 42
        assert splits["policy"] == "iid"
        assert "sizes" in splits


    def test_get_function_definitions(self, splitter):
        """Test OpenAI function definitions."""
        defs = splitter.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) == 1

        func_def = defs[0]
        assert func_def["type"] == "function"
        assert "function" in func_def

        func = func_def["function"]
        assert func["name"] == "Splitter_make_splits"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert "df_ref" in params["properties"]
        assert "target" in params["properties"]
        assert "policy" in params["properties"]
        assert "test_size" in params["properties"]
        assert "val_size" in params["properties"]
        assert "seed" in params["properties"]

        # Check required fields
        assert "df_ref" in params["required"]
        assert "target" in params["required"]
        assert "policy" in params["required"]


class TestFeaturizer:
    """Test Featurizer functionality."""

    @pytest.fixture
    def featurizer(self, tmp_path):
        """Create a Featurizer instance."""
        return Featurizer(tmp_path / "artifacts")

    @pytest.fixture
    def mixed_df_ref(self, tmp_path):
        """Create a dataset with mixed types."""
        np.random.seed(42)
        df = pd.DataFrame({
            "numeric1": np.random.randn(50),
            "numeric2": np.random.randn(50),
            "categorical": np.random.choice(["A", "B", "C"], 50),
            "high_cardinality": [f"cat_{i}" for i in range(50)],
            "target": np.random.choice([0, 1], 50)
        })

        df_path = tmp_path / "mixed_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    @pytest.fixture
    def df_with_missing_ref(self, tmp_path):
        """Create a dataset with missing values."""
        np.random.seed(42)
        df = pd.DataFrame({
            "numeric_with_missing": [1.0, 2.0, np.nan, 4.0, 5.0] * 10,
            "categorical_with_missing": ["A", "B", None, "C", "D"] * 10,
            "complete_numeric": np.random.randn(50),
            "target": np.random.choice([0, 1], 50)
        })

        df_path = tmp_path / "missing_data.pkl"
        save_pickle(df, df_path)
        return str(df_path)

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        artifacts_dir = tmp_path / "artifacts"
        featurizer = Featurizer(artifacts_dir)

        assert featurizer.artifacts_dir == artifacts_dir
        assert featurizer.handoffs_dir.exists()
        assert featurizer.data_dir.exists()

    def test_plan_basic_structure(self, featurizer, mixed_df_ref):
        """Test basic feature plan structure."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        assert "plan_ref" in result
        assert "hash" in result
        assert "summary" in result

        # Load the actual plan
        plan = load_json(result["plan_ref"])

        assert "target" in plan
        assert "feature_columns" in plan
        assert "denied_columns" in plan
        assert "transforms" in plan
        assert "rationale" in plan

    def test_plan_excludes_target(self, featurizer, mixed_df_ref):
        """Test that target column is excluded from features."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        plan = load_json(result["plan_ref"])

        assert plan["target"] == "target"
        assert "target" not in plan["feature_columns"]

    def test_plan_with_deny_list(self, featurizer, mixed_df_ref):
        """Test feature plan with deny list."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target",
            deny_list=["numeric1", "categorical"]
        )

        plan = load_json(result["plan_ref"])

        assert "numeric1" not in plan["feature_columns"]
        assert "categorical" not in plan["feature_columns"]
        assert plan["denied_columns"] == ["numeric1", "categorical"]

    def test_plan_transform_detection(self, featurizer, mixed_df_ref):
        """Test that appropriate transforms are detected."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        plan = load_json(result["plan_ref"])

        # Check transforms for each column type
        for col_info in plan["transforms"]:
            if col_info["column"] == "numeric1":
                assert "standard_scale" in col_info["transforms"]
            elif col_info["column"] == "categorical":
                assert "label_encode" in col_info["transforms"]
            elif col_info["column"] == "high_cardinality":
                # 50 unique values, threshold is >50, so it will be label_encode
                assert "label_encode" in col_info["transforms"] or "high_cardinality_encode" in col_info["transforms"]

    def test_plan_missing_value_detection(self, featurizer, df_with_missing_ref):
        """Test detection of missing values and imputation strategies."""
        result = featurizer.plan(
            df_ref=df_with_missing_ref,
            target="target"
        )

        plan = load_json(result["plan_ref"])

        # Check imputation strategies
        for col_info in plan["transforms"]:
            if col_info["column"] == "numeric_with_missing":
                assert col_info["missing_count"] > 0
                assert "impute_median" in col_info["transforms"]
            elif col_info["column"] == "categorical_with_missing":
                assert col_info["missing_count"] > 0
                assert "impute_mode" in col_info["transforms"]

    def test_plan_rationale(self, featurizer, mixed_df_ref):
        """Test that rationale is included in plan."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        plan = load_json(result["plan_ref"])
        rationale = plan["rationale"]

        assert "total_features" in rationale
        assert "excluded_features" in rationale
        assert "numeric_features" in rationale
        assert "categorical_features" in rationale
        assert "missing_data_columns" in rationale

        # Check counts
        assert rationale["total_features"] == 4  # All columns except target
        assert rationale["excluded_features"] == 0  # No deny list
        assert rationale["numeric_features"] == 2  # numeric1, numeric2
        assert rationale["categorical_features"] == 2  # categorical, high_cardinality

    def test_plan_summary(self, featurizer, mixed_df_ref):
        """Test summary information in result."""
        result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        summary = result["summary"]
        assert "total_features" in summary
        assert "numeric_features" in summary
        assert "categorical_features" in summary

    def test_get_function_definitions(self, featurizer):
        """Test that function definitions are returned."""
        defs = featurizer.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check that both plan and apply are included
        func_names = [d["function"]["name"] for d in defs]
        assert "Featurizer_plan" in func_names
        assert "Featurizer_apply" in func_names

    def test_apply_basic(self, featurizer, mixed_df_ref, tmp_path):
        """Test basic apply functionality."""
        # Create a plan
        plan_result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        # Create splits
        from botds.tools.features import Splitter
        splitter = Splitter(artifacts_dir=tmp_path)
        splits_result = splitter.make_splits(
            df_ref=mixed_df_ref,
            target="target",
            policy="iid"
        )

        # Apply the plan
        result = featurizer.apply(
            df_ref=mixed_df_ref,
            plan_ref=plan_result["plan_ref"],
            splits_ref=splits_result["splits_ref"]
        )

        # Check result structure
        assert "X_train_ref" in result
        assert "y_train_ref" in result
        assert "X_val_ref" in result
        assert "y_val_ref" in result
        assert "X_test_ref" in result
        assert "y_test_ref" in result

    def test_apply_creates_matrices(self, featurizer, mixed_df_ref, tmp_path):
        """Test that apply creates proper train/val/test matrices."""
        # Create a plan
        plan_result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        # Create splits
        from botds.tools.features import Splitter
        splitter = Splitter(artifacts_dir=tmp_path)
        splits_result = splitter.make_splits(
            df_ref=mixed_df_ref,
            target="target",
            policy="iid"
        )

        # Apply the plan
        result = featurizer.apply(
            df_ref=mixed_df_ref,
            plan_ref=plan_result["plan_ref"],
            splits_ref=splits_result["splits_ref"]
        )

        # Load and verify matrices
        X_train = load_pickle(result["X_train_ref"])
        y_train = load_pickle(result["y_train_ref"])
        X_test = load_pickle(result["X_test_ref"])
        y_test = load_pickle(result["y_test_ref"])

        # Check shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]  # Same features

    def test_apply_separates_target(self, featurizer, mixed_df_ref, tmp_path):
        """Test that apply properly separates target from features."""
        # Create a plan
        plan_result = featurizer.plan(
            df_ref=mixed_df_ref,
            target="target"
        )

        # Create splits
        from botds.tools.features import Splitter
        splitter = Splitter(artifacts_dir=tmp_path)
        splits_result = splitter.make_splits(
            df_ref=mixed_df_ref,
            target="target",
            policy="iid"
        )

        # Apply the plan
        result = featurizer.apply(
            df_ref=mixed_df_ref,
            plan_ref=plan_result["plan_ref"],
            splits_ref=splits_result["splits_ref"]
        )

        # Load matrices
        X_train = load_pickle(result["X_train_ref"])
        y_train = load_pickle(result["y_train_ref"])

        # Target should not be in features
        assert "target" not in X_train.columns

        # y should be a Series
        assert isinstance(y_train, pd.Series)

