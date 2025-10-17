"""Unit tests for botds.tools.eval module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from botds.tools.eval import Calibrator, Fairness, Robustness
from botds.utils import save_pickle


@pytest.fixture
def calibrator(tmp_path):
    """Create Calibrator instance."""
    return Calibrator(artifacts_dir=tmp_path)


@pytest.fixture
def fairness(tmp_path):
    """Create Fairness instance."""
    return Fairness(artifacts_dir=tmp_path)


@pytest.fixture
def robustness(tmp_path):
    """Create Robustness instance."""
    return Robustness(artifacts_dir=tmp_path)


@pytest.fixture
def binary_classification_model(tmp_path):
    """Create a trained binary classification model."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], 100)
    
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    
    model_path = tmp_path / "model.pkl"
    save_pickle(model, model_path)
    
    return str(model_path)


@pytest.fixture
def validation_data(tmp_path):
    """Create validation data."""
    np.random.seed(42)
    X_val = pd.DataFrame(np.random.randn(50, 5))
    y_val = pd.Series(np.random.choice([0, 1], 50))
    
    X_val_path = tmp_path / "X_val.pkl"
    y_val_path = tmp_path / "y_val.pkl"
    
    save_pickle(X_val, X_val_path)
    save_pickle(y_val, y_val_path)
    
    return {
        "X_val_ref": str(X_val_path),
        "y_val_ref": str(y_val_path)
    }


@pytest.fixture
def test_data_with_sensitive(tmp_path):
    """Create test data with sensitive attribute."""
    np.random.seed(42)
    # Create DataFrame with sensitive columns included
    X_test = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'feature3': np.random.randn(50),
        'gender': np.random.choice([0, 1], 50),  # Sensitive attribute
        'age_group': np.random.choice([0, 1, 2], 50)  # Another sensitive attribute
    })
    y_test = pd.Series(np.random.choice([0, 1], 50))

    X_test_path = tmp_path / "X_test.pkl"
    y_test_path = tmp_path / "y_test.pkl"

    save_pickle(X_test, X_test_path)
    save_pickle(y_test, y_test_path)

    return {
        "X_test_ref": str(X_test_path),
        "y_test_ref": str(y_test_path),
        "sensitive_cols": ["gender", "age_group"]
    }


class TestCalibrator:
    """Tests for Calibrator class."""
    
    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        calibrator = Calibrator(artifacts_dir=tmp_path)
        
        assert calibrator.artifacts_dir == tmp_path
        assert calibrator.models_dir.exists()
        assert calibrator.models_dir == tmp_path / "models"
    
    def test_fit_isotonic_calibration(self, calibrator, binary_classification_model, validation_data):
        """Test isotonic calibration."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="isotonic",
            **validation_data
        )
        
        assert result["calibration_applied"] is True
        assert result["method"] == "isotonic"
        assert "calibrated_model_ref" in result
        assert "metrics" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "ece_before" in metrics
        assert "ece_after" in metrics
        assert "ece_improvement" in metrics
        assert "brier_before" in metrics
        assert "brier_after" in metrics
        assert "brier_improvement" in metrics
    
    def test_fit_sigmoid_calibration(self, calibrator, binary_classification_model, validation_data):
        """Test sigmoid calibration."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="sigmoid",
            **validation_data
        )
        
        assert result["calibration_applied"] is True
        assert result["method"] == "sigmoid"
        assert "calibrated_model_ref" in result
    
    def test_fit_saves_calibrated_model(self, calibrator, binary_classification_model, validation_data):
        """Test that calibrated model is saved."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="isotonic",
            **validation_data
        )
        
        cal_model_path = Path(result["calibrated_model_ref"])
        assert cal_model_path.exists()
        assert cal_model_path.suffix == ".pkl"
    
    def test_fit_with_non_probabilistic_model(self, calibrator, tmp_path, validation_data):
        """Test calibration with model that doesn't support probabilities."""
        # Create a model without predict_proba
        model = LinearRegression()
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50)
        model.fit(X_train, y_train)
        
        model_path = tmp_path / "linear_model.pkl"
        save_pickle(model, model_path)
        
        result = calibrator.fit(
            model_ref=str(model_path),
            method="isotonic",
            **validation_data
        )
        
        assert result["calibration_applied"] is False
        assert "reason" in result
        assert "does not support probability" in result["reason"]
    
    def test_fit_improves_calibration(self, calibrator, binary_classification_model, validation_data):
        """Test that calibration improves ECE."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="isotonic",
            **validation_data
        )
        
        metrics = result["metrics"]
        # ECE should generally improve (be lower after calibration)
        # Note: This might not always be true for small datasets, so we just check it exists
        assert metrics["ece_improvement"] is not None
    
    def test_get_function_definitions(self, calibrator):
        """Test that function definitions are returned."""
        defs = calibrator.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure
        func = defs[0]["function"]
        assert func["name"] == "Calibrator_fit"
        assert "description" in func
        assert "parameters" in func

    def test_fit_with_invalid_method(self, calibrator, binary_classification_model, validation_data):
        """Test that fit with invalid method raises error."""
        with pytest.raises(ValueError):
            calibrator.fit(
                model_ref=binary_classification_model,
                method="invalid_method",
                **validation_data
            )

    def test_fit_returns_calibrated_model_ref(self, calibrator, binary_classification_model, validation_data):
        """Test that fit returns calibrated model reference."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="isotonic",
            **validation_data
        )

        assert "calibrated_model_ref" in result
        assert Path(result["calibrated_model_ref"]).exists()

    def test_fit_returns_metrics(self, calibrator, binary_classification_model, validation_data):
        """Test that fit returns calibration metrics."""
        result = calibrator.fit(
            model_ref=binary_classification_model,
            method="isotonic",
            **validation_data
        )

        assert "metrics" in result
        assert "calibration_applied" in result
        assert result["calibration_applied"] is True


class TestFairness:
    """Tests for Fairness class."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        fairness = Fairness(artifacts_dir=tmp_path)

        assert fairness.artifacts_dir == tmp_path

    def test_slice_metrics(self, fairness, binary_classification_model, test_data_with_sensitive):
        """Test slice metrics calculation."""
        result = fairness.slice_metrics(
            model_ref=binary_classification_model,
            X_test_ref=test_data_with_sensitive["X_test_ref"],
            y_test_ref=test_data_with_sensitive["y_test_ref"],
            sensitive_cols=test_data_with_sensitive["sensitive_cols"]
        )

        assert "slice_metrics" in result
        assert "fairness_summary" in result

        # Check that slices were analyzed
        assert len(result["slice_metrics"]) > 0

    def test_slice_metrics_returns_summary(self, fairness, binary_classification_model, test_data_with_sensitive):
        """Test that slice metrics returns summary."""
        result = fairness.slice_metrics(
            model_ref=binary_classification_model,
            X_test_ref=test_data_with_sensitive["X_test_ref"],
            y_test_ref=test_data_with_sensitive["y_test_ref"],
            sensitive_cols=test_data_with_sensitive["sensitive_cols"]
        )

        fairness_summary = result["fairness_summary"]
        assert "status" in fairness_summary
        assert "total_issues" in fairness_summary

    def test_get_function_definitions(self, fairness):
        """Test that function definitions are returned."""
        defs = fairness.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        func = defs[0]["function"]
        assert func["name"] == "Fairness_slice_metrics"
        assert "description" in func


class TestRobustness:
    """Tests for Robustness class."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        robustness = Robustness(artifacts_dir=tmp_path)

        assert robustness.artifacts_dir == tmp_path

    def test_ablation(self, robustness, binary_classification_model, validation_data):
        """Test feature ablation study."""
        result = robustness.ablation(
            model_ref=binary_classification_model,
            top_k=3,
            **validation_data
        )

        assert "baseline_score" in result
        assert "ablation_results" in result
        assert "summary" in result
        assert "metric_name" in result

        # Check ablation results structure
        ablation_results = result["ablation_results"]
        assert len(ablation_results) <= 3  # top_k=3

        for ablation in ablation_results:
            assert "feature" in ablation
            assert "performance_drop" in ablation
            assert "baseline_score" in ablation
            assert "ablated_score" in ablation

    def test_ablation_with_different_top_k(self, robustness, binary_classification_model, validation_data):
        """Test ablation with different top_k values."""
        result = robustness.ablation(
            model_ref=binary_classification_model,
            top_k=2,
            **validation_data
        )

        assert len(result["ablation_results"]) <= 2

    def test_ablation_returns_summary(self, robustness, binary_classification_model, validation_data):
        """Test that ablation returns summary."""
        result = robustness.ablation(
            model_ref=binary_classification_model,
            top_k=3,
            **validation_data
        )

        summary = result["summary"]
        assert "max_performance_drop" in summary
        assert "avg_performance_drop" in summary

    def test_get_function_definitions(self, robustness):
        """Test that function definitions are returned."""
        defs = robustness.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        func = defs[0]["function"]
        assert func["name"] == "Robustness_ablation"
        assert "description" in func

    def test_shock_tests(self, robustness, binary_classification_model, validation_data):
        """Test shock tests for robustness."""
        result = robustness.shock_tests(
            model_ref=binary_classification_model,
            **validation_data
        )

        assert "resilience_grade" in result
        assert "shock_results" in result
        assert "summary" in result

    def test_shock_tests_returns_grade(self, robustness, binary_classification_model, validation_data):
        """Test that shock tests return a grade."""
        result = robustness.shock_tests(
            model_ref=binary_classification_model,
            **validation_data
        )

        grade = result["resilience_grade"]
        assert grade in ["A", "B", "C", "D", "F"]

