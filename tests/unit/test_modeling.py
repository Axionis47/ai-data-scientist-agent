"""Unit tests for botds.tools.modeling module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from botds.tools.modeling import ModelTrainer, Tuner
from botds.utils import save_pickle, load_pickle


@pytest.fixture
def trainer(tmp_path):
    """Create ModelTrainer instance."""
    return ModelTrainer(artifacts_dir=tmp_path)


@pytest.fixture
def classification_data(tmp_path):
    """Create classification training data."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.choice([0, 1], 100))
    X_val = pd.DataFrame(np.random.randn(20, 5), columns=[f"feat_{i}" for i in range(5)])
    y_val = pd.Series(np.random.choice([0, 1], 20))

    # Save to files
    X_train_ref = tmp_path / "X_train.pkl"
    y_train_ref = tmp_path / "y_train.pkl"
    X_val_ref = tmp_path / "X_val.pkl"
    y_val_ref = tmp_path / "y_val.pkl"

    save_pickle(X_train, X_train_ref)
    save_pickle(y_train, y_train_ref)
    save_pickle(X_val, X_val_ref)
    save_pickle(y_val, y_val_ref)

    return {
        "X_train_ref": str(X_train_ref),
        "y_train_ref": str(y_train_ref),
        "X_val_ref": str(X_val_ref),
        "y_val_ref": str(y_val_ref)
    }


@pytest.fixture
def regression_data(tmp_path):
    """Create regression training data."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.randn(100))
    X_val = pd.DataFrame(np.random.randn(20, 5), columns=[f"feat_{i}" for i in range(5)])
    y_val = pd.Series(np.random.randn(20))

    # Save to files
    X_train_ref = tmp_path / "X_train_reg.pkl"
    y_train_ref = tmp_path / "y_train_reg.pkl"
    X_val_ref = tmp_path / "X_val_reg.pkl"
    y_val_ref = tmp_path / "y_val_reg.pkl"

    save_pickle(X_train, X_train_ref)
    save_pickle(y_train, y_train_ref)
    save_pickle(X_val, X_val_ref)
    save_pickle(y_val, y_val_ref)

    return {
        "X_train_ref": str(X_train_ref),
        "y_train_ref": str(y_train_ref),
        "X_val_ref": str(X_val_ref),
        "y_val_ref": str(y_val_ref)
    }


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        trainer = ModelTrainer(artifacts_dir=tmp_path)

        assert trainer.artifacts_dir == tmp_path
        assert trainer.models_dir.exists()
        assert trainer.models_dir == tmp_path / "models"

    def test_train_logistic_regression(self, trainer, classification_data):
        """Test training logistic regression model."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        # Check result structure
        assert "model_ref" in result
        assert "model_id" in result
        assert "model_spec" in result
        assert "val_metrics" in result
        assert "train_time_seconds" in result
        assert "task_type" in result

        # Check task type
        assert result["task_type"] == "classification"

        # Check metrics
        assert "accuracy" in result["val_metrics"]
        assert "f1_score" in result["val_metrics"]

        # Check model file exists
        assert Path(result["model_ref"]).exists()

    def test_train_random_forest_classifier(self, trainer, classification_data):
        """Test training random forest classifier."""
        model_spec = {
            "name": "random_forest_classifier",
            "params": {"n_estimators": 10, "max_depth": 3}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        assert result["task_type"] == "classification"
        assert "accuracy" in result["val_metrics"]
        assert result["train_time_seconds"] > 0

    def test_train_linear_regression(self, trainer, regression_data):
        """Test training linear regression model."""
        model_spec = {
            "name": "linear_regression",
            "params": {}
        }

        result = trainer.train(
            model_spec=model_spec,
            **regression_data
        )

        # Check task type
        assert result["task_type"] == "regression"

        # Check metrics
        assert "mae" in result["val_metrics"]
        assert "rmse" in result["val_metrics"]

        # Check model file exists
        assert Path(result["model_ref"]).exists()

    def test_train_random_forest_regressor(self, trainer, regression_data):
        """Test training random forest regressor."""
        model_spec = {
            "name": "random_forest_regressor",
            "params": {"n_estimators": 10, "max_depth": 3}
        }

        result = trainer.train(
            model_spec=model_spec,
            **regression_data
        )

        assert result["task_type"] == "regression"
        assert "mae" in result["val_metrics"]
        assert "rmse" in result["val_metrics"]
        assert result["train_time_seconds"] > 0

    def test_train_time_measured(self, trainer, classification_data):
        """Test that training time is measured."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        assert result["train_time_seconds"] > 0
        assert isinstance(result["train_time_seconds"], float)

    def test_model_id_generation(self, trainer, classification_data):
        """Test that model ID is generated."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        assert "model_id" in result
        assert len(result["model_id"]) <= 16
        assert result["model_id"].startswith("logistic_regress")

    def test_model_saved_to_file(self, trainer, classification_data):
        """Test that model is saved to file."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        model_path = Path(result["model_ref"])
        assert model_path.exists()
        assert model_path.parent == trainer.models_dir
        assert model_path.suffix == ".pkl"

    def test_different_models_different_ids(self, trainer, classification_data):
        """Test that different models get different IDs."""
        model_spec1 = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        model_spec2 = {
            "name": "random_forest_classifier",
            "params": {"n_estimators": 10}
        }

        result1 = trainer.train(model_spec=model_spec1, **classification_data)
        result2 = trainer.train(model_spec=model_spec2, **classification_data)

        assert result1["model_id"] != result2["model_id"]

    def test_validation_metrics_reasonable(self, trainer, classification_data):
        """Test that validation metrics are in reasonable ranges."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        metrics = result["val_metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_regression_metrics_reasonable(self, trainer, regression_data):
        """Test that regression metrics are reasonable."""
        model_spec = {
            "name": "linear_regression",
            "params": {}
        }

        result = trainer.train(
            model_spec=model_spec,
            **regression_data
        )

        metrics = result["val_metrics"]
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_get_function_definitions(self, trainer):
        """Test that function definitions are returned."""
        defs = trainer.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure of first definition
        assert "type" in defs[0]
        assert defs[0]["type"] == "function"
        assert "function" in defs[0]

        # Check function details
        func = defs[0]["function"]
        assert func["name"] == "ModelTrainer_train"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert "model_spec" in params["properties"]
        assert "X_train_ref" in params["properties"]
        assert "y_train_ref" in params["properties"]
        assert "X_val_ref" in params["properties"]
        assert "y_val_ref" in params["properties"]

    def test_unknown_model_raises_error(self, trainer, classification_data):
        """Test that unknown model name raises error."""
        model_spec = {
            "name": "unknown_model",
            "params": {}
        }

        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train(
                model_spec=model_spec,
                **classification_data
            )

    def test_model_with_no_params(self, trainer, classification_data):
        """Test training model with no params specified."""
        model_spec = {
            "name": "logistic_regression"
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        assert "model_ref" in result
        assert result["task_type"] == "classification"

    def test_model_can_be_loaded(self, trainer, classification_data):
        """Test that saved model can be loaded and used."""
        model_spec = {
            "name": "logistic_regression",
            "params": {"max_iter": 100}
        }

        result = trainer.train(
            model_spec=model_spec,
            **classification_data
        )

        # Load the model
        model = load_pickle(result["model_ref"])

        # Make predictions
        X_val = load_pickle(classification_data["X_val_ref"])
        predictions = model.predict(X_val)

        assert len(predictions) == len(X_val)

    def test_multiclass_classification(self, trainer, tmp_path):
        """Test training with multiclass classification."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.choice([0, 1, 2], 100))  # 3 classes
        X_val = pd.DataFrame(np.random.randn(20, 5))
        y_val = pd.Series(np.random.choice([0, 1, 2], 20))

        # Save data
        X_train_ref = tmp_path / "X_train_multi.pkl"
        y_train_ref = tmp_path / "y_train_multi.pkl"
        X_val_ref = tmp_path / "X_val_multi.pkl"
        y_val_ref = tmp_path / "y_val_multi.pkl"

        save_pickle(X_train, X_train_ref)
        save_pickle(y_train, y_train_ref)
        save_pickle(X_val, X_val_ref)
        save_pickle(y_val, y_val_ref)

        model_spec = {
            "name": "random_forest_classifier",
            "params": {"n_estimators": 10}
        }

        result = trainer.train(
            model_spec=model_spec,
            X_train_ref=str(X_train_ref),
            y_train_ref=str(y_train_ref),
            X_val_ref=str(X_val_ref),
            y_val_ref=str(y_val_ref)
        )

        assert result["task_type"] == "classification"
        assert "accuracy" in result["val_metrics"]
        assert "f1_score" in result["val_metrics"]


class TestTuner:
    """Tests for Tuner class."""

    @pytest.fixture
    def tuner(self, tmp_path):
        """Create Tuner instance."""
        return Tuner(artifacts_dir=tmp_path)

    def test_init(self, tmp_path):
        """Test Tuner initialization."""
        tuner = Tuner(artifacts_dir=tmp_path)
        assert tuner.artifacts_dir == tmp_path

    def test_quick_search_random_forest(self, tuner, tmp_path):
        """Test quick search for random forest."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = pd.Series(np.random.choice([0, 1], 50))

        X_train_ref = tmp_path / "X_train_tune.pkl"
        y_train_ref = tmp_path / "y_train_tune.pkl"

        save_pickle(X_train, X_train_ref)
        save_pickle(y_train, y_train_ref)

        result = tuner.quick_search(
            model_family="random_forest",
            X_train_ref=str(X_train_ref),
            y_train_ref=str(y_train_ref),
            budget_minutes=1
        )

        assert "best_spec" in result
        assert "trials" in result
        assert "best_score" in result

        # Check best spec structure
        assert "name" in result["best_spec"]
        assert "params" in result["best_spec"]

    def test_quick_search_logistic_regression(self, tuner, tmp_path):
        """Test quick search for logistic regression."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = pd.Series(np.random.choice([0, 1], 50))

        X_train_ref = tmp_path / "X_train_lr.pkl"
        y_train_ref = tmp_path / "y_train_lr.pkl"

        save_pickle(X_train, X_train_ref)
        save_pickle(y_train, y_train_ref)

        result = tuner.quick_search(
            model_family="logistic_regression",
            X_train_ref=str(X_train_ref),
            y_train_ref=str(y_train_ref),
            budget_minutes=1
        )

        assert "best_spec" in result
        # Tuner adds _classifier suffix for classification tasks
        assert result["best_spec"]["name"] == "logistic_regression_classifier"

    def test_quick_search_trials_recorded(self, tuner, tmp_path):
        """Test that trials are recorded."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = pd.Series(np.random.choice([0, 1], 50))

        X_train_ref = tmp_path / "X_train_trials.pkl"
        y_train_ref = tmp_path / "y_train_trials.pkl"

        save_pickle(X_train, X_train_ref)
        save_pickle(y_train, y_train_ref)

        result = tuner.quick_search(
            model_family="random_forest",
            X_train_ref=str(X_train_ref),
            y_train_ref=str(y_train_ref),
            budget_minutes=1
        )

        assert len(result["trials"]) > 0

        # Check trial structure
        trial = result["trials"][0]
        assert "params" in trial
        assert "score" in trial

    def test_get_function_definitions(self, tuner):
        """Test that function definitions are returned."""
        defs = tuner.get_function_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure
        func = defs[0]["function"]
        assert func["name"] == "Tuner_quick_search"
        assert "description" in func
        assert "parameters" in func


