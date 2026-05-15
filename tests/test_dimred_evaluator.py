"""Unit tests for core/dimred_evaluator.py module.

Tests the dimensionality reduction evaluation framework including nested CV
and statistical significance testing.
"""

import pytest
import numpy as np
import pandas as pd
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
            n_folds=5,
            n_repeats=2,
            random_state=42
        )
        
        assert evaluator.dimred_config == dimred_config
        assert evaluator.random_state == 42
        assert evaluator.logger is not None
        assert evaluator.n_folds == 5
        assert evaluator.n_repeats == 2
    
    def test_model_benefits_from_dimred_linear(self):
        """Test _model_benefits_from_dimred logic for linear models."""
        preprocessor = DataPreprocessor()
        config = DimRedConfig()
        evaluator = DimRedEvaluator(preprocessor, config)
        
        # Linear models should benefit
        assert evaluator._model_benefits_from_dimred('LogisticRegression') is True
        assert evaluator._model_benefits_from_dimred('LinearSVM') is True
        assert evaluator._model_benefits_from_dimred('KNN') is True
    
    def test_model_benefits_from_dimred_tree(self):
        """Test _model_benefits_from_dimred logic for tree models."""
        preprocessor = DataPreprocessor()
        config = DimRedConfig()
        evaluator = DimRedEvaluator(preprocessor, config)
        
        # Tree-based models typically don't benefit as much
        assert evaluator._model_benefits_from_dimred('RandomForest') is False
        assert evaluator._model_benefits_from_dimred('XGBoost') is False
        assert evaluator._model_benefits_from_dimred('GradientBoosting') is False
    
    def test_model_benefits_from_dimred_unknown(self):
        """Test _model_benefits_from_dimred logic for unknown models."""
        preprocessor = DataPreprocessor()
        config = DimRedConfig()
        evaluator = DimRedEvaluator(preprocessor, config)
        
        # Unknown model defaults to True
        assert evaluator._model_benefits_from_dimred('UnknownModel') is True
    
    def test_evaluate_classification_with_dimred(self):
        """Test evaluate_classification_with_dimred for classification."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca'
            }
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=3,
            n_repeats=1,
            random_state=42
        )
        
        # Create sample data as DataFrame for preprocessor
        X, y = make_classification(
            n_samples=100, 
            n_features=20, 
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        result = evaluator.evaluate_classification_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # Should have results for the model
        assert 'LogisticRegression' in result
        model_result = result['LogisticRegression']
        
        # Check basic structure
        assert 'selected_variant' in model_result
        assert 'accuracy_mean' in model_result
        assert 'accuracy_scores' in model_result
        assert isinstance(model_result['accuracy_scores'], list)
    
    def test_evaluate_classification_skip_tree_models(self):
        """Test that tree models are skipped in auto mode."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto'
            }
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=3,
            n_repeats=1,
            random_state=42
        )
        
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=5, random_state=42)
        }
        
        result = evaluator.evaluate_classification_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # Tree model should have been evaluated (just without dimred comparison)
        assert 'RandomForest' in result


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_high_dimensional_classification(self):
        """Test evaluation with high-dimensional data."""
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
            n_folds=3,
            n_repeats=1,
            random_state=42
        )
        
        # High-dimensional dataset
        X, y = make_classification(
            n_samples=100,
            n_features=100,
            n_informative=20,
            n_redundant=20,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        results = evaluator.evaluate_classification_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # Should have results
        assert 'LogisticRegression' in results
        assert results['LogisticRegression']['accuracy_mean'] >= 0.0
    
    def test_low_dimensional_data(self):
        """Test evaluation with low-dimensional data."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto'
            }
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()
        evaluator = DimRedEvaluator(
            preprocessor=preprocessor,
            dimred_config=dimred_config,
            n_folds=3,
            n_repeats=1,
            random_state=42
        )
        
        # Low-dimensional dataset
        X, y = make_classification(
            n_samples=100,
            n_features=5,  # Very few features
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        results = evaluator.evaluate_classification_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # Should have results
        assert 'LogisticRegression' in results
        assert results['LogisticRegression']['accuracy_mean'] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
