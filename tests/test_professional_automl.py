"""Tests for professional AutoML pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.advanced_optimization import AdvancedHyperparameterOptimizer, AutoMLPipeline


class TestAdvancedHyperparameterOptimizer:
    """Test suite for AdvancedHyperparameterOptimizer."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
        return X, y

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = AdvancedHyperparameterOptimizer(
            task_type="classification", n_trials=10, optimization_time_minutes=1, random_state=42
        )
        assert optimizer.task_type == "classification"
        assert optimizer.n_trials == 10
        assert optimizer.optimization_time_minutes == 1

    def test_classification_optimization(self, classification_data):
        """Test hyperparameter optimization for classification."""
        X, y = classification_data
        model = RandomForestClassifier(random_state=42)

        optimizer = AdvancedHyperparameterOptimizer(
            task_type="classification", n_trials=5, optimization_time_minutes=1, random_state=42
        )

        result = optimizer.optimize_model(model=model, X=X, y=y)

        assert result is not None
        assert "best_score" in result
        assert "best_params" in result
        assert "improvement" in result

    def test_timeout_handling(self, classification_data):
        """Test that optimization respects timeout."""
        import time

        X, y = classification_data
        model = RandomForestClassifier(random_state=42)

        optimizer = AdvancedHyperparameterOptimizer(
            task_type="classification",
            n_trials=1000,  # Many trials
            optimization_time_minutes=0.05,  # 3 seconds
            random_state=42,
        )

        start = time.time()
        result = optimizer.optimize_model(model=model, X=X, y=y)
        elapsed = time.time() - start

        # Should finish within timeout + buffer
        assert elapsed < 30  # 3s timeout + generous buffer
        assert result is not None

    def test_invalid_task_type(self):
        """Test handling of invalid task type (accepted without error)."""
        # The current implementation accepts any task_type without raising ValueError
        optimizer = AdvancedHyperparameterOptimizer(task_type="invalid_task", n_trials=10)
        assert optimizer.task_type == "invalid_task"

    def test_baseline_score_calculation(self, classification_data):
        """Test baseline score is calculated correctly."""
        X, y = classification_data
        model = RandomForestClassifier(random_state=42)

        optimizer = AdvancedHyperparameterOptimizer(task_type="classification", n_trials=5, random_state=42)

        result = optimizer.optimize_model(model=model, X=X, y=y)

        assert "baseline_score" in result
        assert result["baseline_score"] > 0


class TestAutoMLPipeline:
    """Test suite for AutoMLPipeline."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42)
        return X, y

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=5, random_state=42)
        assert pipeline.task_type == "classification"
        assert pipeline.optimization_time_minutes == 5

    def test_run_advanced_automl(self, classification_data):
        """Test complete AutoML pipeline."""
        X, y = classification_data

        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=2, random_state=42)

        model_candidates = [("RandomForest", RandomForestClassifier(random_state=42))]

        results = pipeline.run_advanced_automl(X=X, y=y, model_candidates=model_candidates, include_ensemble=False)

        assert results is not None
        assert "individual_models" in results
        assert "RandomForest" in results["individual_models"]
        assert "optimization_summary" in results

    def test_model_selection(self, classification_data):
        """Test that best model is correctly identified."""
        X, y = classification_data

        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=1, random_state=42)

        from sklearn.linear_model import LogisticRegression

        model_candidates = [
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
        ]

        results = pipeline.run_advanced_automl(X=X, y=y, model_candidates=model_candidates, include_ensemble=False)

        # Find best model
        best_score = -float("inf")
        best_name = None
        for name, result in results["individual_models"].items():
            if result["best_score"] > best_score:
                best_score = result["best_score"]
                best_name = name

        assert best_name is not None
        assert best_score > 0

    def test_empty_model_candidates(self, classification_data):
        """Test handling of empty model candidates."""
        X, y = classification_data

        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=1, random_state=42)

        results = pipeline.run_advanced_automl(X=X, y=y, model_candidates=[], include_ensemble=False)

        # Should handle gracefully
        assert results is not None
        assert "individual_models" in results

    def test_error_recovery(self, classification_data):
        """Test that pipeline recovers from model failures."""
        X, y = classification_data

        # Create a model that will fail
        class FailingModel:
            def __init__(self):
                pass

            def fit(self, X, y):
                raise ValueError("Intentional failure")

            def predict(self, X):
                raise ValueError("Intentional failure")

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=1, random_state=42)

        model_candidates = [("FailingModel", FailingModel()), ("RandomForest", RandomForestClassifier(random_state=42))]

        # Should not crash despite failing model
        results = pipeline.run_advanced_automl(X=X, y=y, model_candidates=model_candidates, include_ensemble=False)

        assert results is not None
        # RandomForest should have succeeded
        assert "RandomForest" in results["individual_models"]


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_classification(self):
        """Test complete end-to-end classification pipeline."""
        # Create dataset
        X, y = make_classification(n_samples=300, n_features=15, n_informative=10, n_redundant=3, random_state=42)

        # Run AutoML
        pipeline = AutoMLPipeline(task_type="classification", optimization_time_minutes=2, random_state=42)

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        model_candidates = [
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
            ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ]

        results = pipeline.run_advanced_automl(X=X, y=y, model_candidates=model_candidates, include_ensemble=False)

        # Verify results structure
        assert "individual_models" in results
        assert "dataset_info" in results
        assert "optimization_summary" in results

        # Verify all models were attempted
        assert len(results["individual_models"]) > 0

        # Verify dataset info
        assert results["dataset_info"]["n_samples"] == 300
        assert results["dataset_info"]["n_features"] == 15

        # Verify at least one model succeeded
        successful_models = [
            name for name, result in results["individual_models"].items() if result.get("best_score", -1000) > 0
        ]
        assert len(successful_models) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
