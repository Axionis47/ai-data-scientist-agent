"""Unit tests for botds.tools.plotter module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression

from botds.tools.plotter import Plotter
from botds.utils import save_pickle


@pytest.fixture
def plotter(tmp_path):
    """Create Plotter instance."""
    return Plotter(artifacts_dir=tmp_path)


@pytest.fixture
def binary_classification_model(tmp_path):
    """Create binary classification model and test data."""
    np.random.seed(42)

    # Create training data
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], 100)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Create test data
    X_test = np.random.randn(50, 5)
    y_test = np.random.choice([0, 1], 50)

    # Save to files
    model_path = tmp_path / "model.pkl"
    X_test_path = tmp_path / "X_test.pkl"
    y_test_path = tmp_path / "y_test.pkl"

    save_pickle(model, model_path)
    save_pickle(X_test, X_test_path)
    save_pickle(y_test, y_test_path)

    return {
        "model_ref": str(model_path),
        "X_test_ref": str(X_test_path),
        "y_test_ref": str(y_test_path)
    }


class TestPlotter:
    """Tests for Plotter class."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        plotter = Plotter(artifacts_dir=tmp_path)

        assert plotter.artifacts_dir == tmp_path
        assert plotter.plots_dir.exists()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_pr_curve(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test PR curve generation."""
        result = plotter.pr_curve(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"]
        )

        assert "plot_ref" in result
        assert "pr_auc" in result

        # Check that matplotlib was called
        assert mock_savefig.called
        assert mock_close.called

        # Check AUC is valid
        assert 0 <= result["pr_auc"] <= 1

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_pr_curve_with_title(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test PR curve with custom title."""
        result = plotter.pr_curve(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"],
            title="Custom PR Curve"
        )

        assert "plot_ref" in result
        assert mock_savefig.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_lift_curve(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test lift curve generation."""
        result = plotter.lift_curve(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"]
        )

        assert "plot_ref" in result

        # Check that matplotlib was called
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_lift_curve_with_title(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test lift curve with custom title."""
        result = plotter.lift_curve(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"],
            title="Custom Lift Curve"
        )

        assert "plot_ref" in result
        assert mock_savefig.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_calibration_plot(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test calibration plot generation."""
        result = plotter.calibration_plot(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"]
        )

        assert "plot_ref" in result

        # Check that matplotlib was called
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_calibration_plot_with_title(self, mock_close, mock_savefig, plotter, binary_classification_model):
        """Test calibration plot with custom title."""
        result = plotter.calibration_plot(
            model_ref=binary_classification_model["model_ref"],
            X_test_ref=binary_classification_model["X_test_ref"],
            y_test_ref=binary_classification_model["y_test_ref"],
            title="Custom Calibration Plot"
        )

        assert "plot_ref" in result
        assert mock_savefig.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_bars(self, mock_close, mock_savefig, plotter):
        """Test bar chart generation."""
        data = {
            "Category A": 10.5,
            "Category B": 20.3,
            "Category C": 15.7
        }

        result = plotter.bars(
            data=data,
            title="Test Bar Chart"
        )

        assert "plot_ref" in result

        # Check that matplotlib was called
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_bars_with_labels(self, mock_close, mock_savefig, plotter):
        """Test bar chart with custom labels."""
        data = {"A": 1.0, "B": 2.0}

        result = plotter.bars(
            data=data,
            title="Test",
            xlabel="X Label",
            ylabel="Y Label"
        )

        assert "plot_ref" in result
        assert mock_savefig.called

    def test_get_function_definitions(self, plotter):
        """Test that function definitions are returned."""
        defs = plotter.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check that plot functions are included
        func_names = [d["function"]["name"] for d in defs]
        assert "Plotter_pr_curve" in func_names
        assert "Plotter_lift_curve" in func_names
        assert "Plotter_calibration_plot" in func_names
        assert "Plotter_bars" in func_names

