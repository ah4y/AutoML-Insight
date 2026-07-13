"""Unit tests for core/dimred_evaluator.py module.

Tests the dimensionality reduction evaluation framework including nested CV
and statistical significance testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from core.dimred_evaluator import DimRedEvaluator
from core.dimred import DimRedConfig
from core.preprocess import DataPreprocessor


class TestDimRedEvaluator:
    """Test DimRedEvaluator class."""
    
    def test_initialization(self):
        """Test DimRedEvaluator initialization."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'pca',
                'variance_target': 0.95
            }
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()
        
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            random_state=42
        )
        
        assert evaluator.dimred_config == dimred_config
        assert evaluator.random_state == 42
        assert evaluator.logger is not None
        assert evaluator.preprocessor is preprocessor
    
    def test_model_benefits_from_dimred(self):
        """Test _model_benefits_from_dimred logic."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        # Linear models should benefit (pass string name, not object)
        assert evaluator._model_benefits_from_dimred("LogisticRegression") is True
        assert evaluator._model_benefits_from_dimred("KNN") is True
        assert evaluator._model_benefits_from_dimred("LinearSVM") is True
        
        # Tree-based models should NOT benefit
        assert evaluator._model_benefits_from_dimred("RandomForest") is False
        assert evaluator._model_benefits_from_dimred("XGBoost") is False
        
        # Unknown model defaults to True
        assert evaluator._model_benefits_from_dimred("UnknownModel") is True
    
    def test_evaluate_classification_with_dimred(self):
        """Test evaluate_classification_with_dimred runs without error."""
        config_dict = {
            'dimred': {
                'enable': 'off',  # Disable for faster test
                'method': 'pca'
            }
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()
        
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=2,
            n_repeats=1,
            random_state=42
        )
        
        # Create sample data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        results = evaluator.evaluate_classification_with_dimred(
            models, X_df, y, task_type="classification"
        )
        
        assert isinstance(results, dict)
        assert 'LogisticRegression' in results
    
    def test_compare_and_select_variant_dimred_better(self):
        """Test _compare_and_select_variant when dimred is better."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        # Baseline results dict
        baseline_results = {
            'accuracy_mean': 0.75,
            'accuracy_scores': [0.72, 0.74, 0.76, 0.73, 0.75],
            'uses_dimred': False
        }
        # Dimred results are much better
        dimred_results = {
            'accuracy_mean': 0.88,
            'accuracy_scores': [0.86, 0.88, 0.89, 0.87, 0.90],
            'uses_dimred': True,
            'dimred_method': 'PCA'
        }
        
        result = evaluator._compare_and_select_variant(
            "TestModel", baseline_results, dimred_results
        )
        
        assert result['selected_variant'] == 'dimred'
        assert 'comparison' in result
        assert isinstance(result['comparison'], dict)
        assert result['comparison']['improvement'] > 0
    
    def test_compare_and_select_variant_baseline_better(self):
        """Test _compare_and_select_variant when baseline is better."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        # Baseline results are better
        baseline_results = {
            'accuracy_mean': 0.90,
            'accuracy_scores': [0.88, 0.90, 0.91, 0.89, 0.92],
            'uses_dimred': False
        }
        dimred_results = {
            'accuracy_mean': 0.75,
            'accuracy_scores': [0.73, 0.75, 0.76, 0.74, 0.77],
            'uses_dimred': True
        }
        
        result = evaluator._compare_and_select_variant(
            "TestModel", baseline_results, dimred_results
        )
        
        assert result['selected_variant'] == 'baseline'
    
    def test_compare_and_select_variant_no_dimred(self):
        """Test _compare_and_select_variant when dimred was not evaluated."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        baseline_results = {
            'accuracy_mean': 0.85,
            'uses_dimred': False
        }
        
        result = evaluator._compare_and_select_variant(
            "TestModel", baseline_results, None
        )
        
        assert result['selected_variant'] == 'baseline'
        assert result['comparison'] == 'dimred_not_evaluated'
    
    def test_get_leaderboard_empty(self):
        """Test get_leaderboard_with_dimred with no results."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        leaderboard = evaluator.get_leaderboard_with_dimred()
        assert isinstance(leaderboard, list)
        assert len(leaderboard) == 0
    
    def test_get_dimred_summary_empty(self):
        """Test get_dimred_summary with no results."""
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(preprocessor=preprocessor)
        
        summary = evaluator.get_dimred_summary()
        assert isinstance(summary, dict)
        assert summary['total_models_evaluated'] == 0


class TestIntegrationScenarios:
    """Integration tests for real-world evaluation scenarios."""
    
    def test_evaluator_with_dataframe(self):
        """Test evaluator with DataFrame input."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        
        preprocessor = DataPreprocessor()
        dimred_config = DimRedConfig(enable='off')  # Disable for speed
        
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=2,
            n_repeats=1,
            random_state=42
        )
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=200)
        }
        
        results = evaluator.evaluate_classification_with_dimred(
            models, X_df, y, task_type="classification"
        )
        
        assert 'LogisticRegression' in results
    
    def test_multiple_models_evaluation(self):
        """Test evaluation with multiple models."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        
        preprocessor = DataPreprocessor()
        dimred_config = DimRedConfig(enable='off')
        
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=2,
            n_repeats=1,
            random_state=42
        )
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        results = evaluator.evaluate_classification_with_dimred(
            models, X_df, y, task_type="classification"
        )
        
        # Should have results for both models
        assert 'LogisticRegression' in results
        assert 'RandomForest' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])