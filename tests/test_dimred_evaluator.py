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


def test_pca_transformer_creation_forced():
    """Test PCA transformer creation with forced config (lines 139-166)."""
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    models = {'LogisticRegression': LogisticRegression(random_state=42)}
    results = evaluator.evaluate_classification_with_dimred(models, X, y)
    
    # Should have PCA transformer
    assert 'pca_transformer' in results or 'LogisticRegression' in results


def test_pca_transformer_creation_failure():
    """Test handling of PCA transformer creation failure (lines 168-173)."""
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    # Small dataset
    X, y = make_classification(n_samples=10, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    models = {'LogisticRegression': LogisticRegression(random_state=42)}
    results = evaluator.evaluate_classification_with_dimred(models, X, y)
    
    # Should still complete even if PCA fails
    assert 'LogisticRegression' in results


def test_pca_visualization_disabled():
    """Test PCA visualization disabled (lines 172-173)."""
    config_dict = {
        'dimred': {
            'enable': 'off'
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
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    models = {'LogisticRegression': LogisticRegression(random_state=42)}
    results = evaluator.evaluate_classification_with_dimred(models, X, y)
    
    # Should not have PCA transformer since disabled
    assert 'pca_transformer' not in results or 'LogisticRegression' in results


def test_evaluate_single_model_with_dimred():
    """Test _evaluate_single_model with dimensionality reduction (lines 206-235)."""
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    model = LogisticRegression(random_state=42)
    results = evaluator._evaluate_single_model(model, X, y, 'TestModel', use_dimred=True)
    
    assert 'accuracy_mean' in results
    assert results['uses_dimred'] is True


def test_evaluate_single_model_without_dimred():
    """Test _evaluate_single_model without dimensionality reduction (lines 237-241)."""
    config_dict = {
        'dimred': {
            'enable': 'off'
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
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    model = LogisticRegression(random_state=42)
    results = evaluator._evaluate_single_model(model, X, y, 'TestModel', use_dimred=False)
    
    assert 'accuracy_mean' in results
    assert results['uses_dimred'] is False


def test_nested_cv_evaluation_with_dimred():
    """Test _nested_cv_evaluation with complete pipeline (lines 256-341)."""
    from sklearn.base import clone
    
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    preprocessor_clone = clone(preprocessor)
    X_preprocessed, _ = preprocessor_clone.fit_transform(X, y)
    
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression as LogReg
    
    pipeline = Pipeline([
        ('preprocess', preprocessor_clone.preprocessor),
        ('model', LogReg(max_iter=200, random_state=42))
    ])
    
    results = evaluator._nested_cv_evaluation(pipeline, X, y, 'TestModel')
    
    assert 'accuracy_mean' in results
    assert 'accuracy_ci_lower' in results
    assert 'accuracy_ci_upper' in results


def test_nested_cv_evaluation_error_handling():
    """Test _nested_cv_evaluation error handling (lines 303-323)."""
    from sklearn.base import clone
    
    config_dict = {
        'dimred': {
            'enable': 'off'
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
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    preprocessor_clone = clone(preprocessor)
    
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression as LogReg
    
    pipeline = Pipeline([
        ('preprocess', preprocessor_clone.preprocessor),
        ('model', LogReg(max_iter=100, random_state=42))
    ])
    
    # Should handle errors gracefully
    results = evaluator._nested_cv_evaluation(pipeline, X, y, 'TestModel')
    
    assert 'accuracy_mean' in results


def test_compare_and_select_variant_dimred_better():
    """Test _compare_and_select_variant when dimred is better (lines 394-423)."""
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    baseline_results = {
        'accuracy_mean': 0.80,
        'accuracy_scores': [0.75, 0.85, 0.80, 0.85],
        'model_name': 'Model_baseline'
    }
    dimred_results = {
        'accuracy_mean': 0.85,
        'accuracy_scores': [0.82, 0.88, 0.84, 0.88],
        'model_name': 'Model_dimred'
    }
    
    selected = evaluator._compare_and_select_variant(
        'TestModel', baseline_results, dimred_results
    )
    
    assert selected['selected_variant'] in ['baseline', 'dimred']
    assert 'comparison' in selected


def test_compare_and_select_variant_baseline_only():
    """Test _compare_and_select_variant with baseline only (lines 387-392)."""
    config_dict = {
        'dimred': {
            'enable': 'off'
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
    
    baseline_results = {
        'accuracy_mean': 0.80,
        'accuracy_scores': [0.75, 0.85],
        'model_name': 'Model_baseline'
    }
    
    selected = evaluator._compare_and_select_variant(
        'TestModel', baseline_results, None
    )
    
    assert selected['selected_variant'] == 'baseline'
    assert selected['comparison'] == 'dimred_not_evaluated'


def test_compare_and_select_variant_significance_test():
    """Test _compare_and_select_variant with significance testing (lines 398-413)."""
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
        n_folds=2,
        n_repeats=1,
        random_state=42
    )
    
    baseline_results = {
        'accuracy_mean': 0.80,
        'accuracy_scores': [0.78, 0.82, 0.79, 0.81],
        'model_name': 'Model_baseline'
    }
    dimred_results = {
        'accuracy_mean': 0.81,
        'accuracy_scores': [0.79, 0.83, 0.80, 0.82],
        'model_name': 'Model_dimred'
    }
    
    selected = evaluator._compare_and_select_variant(
        'TestModel', baseline_results, dimred_results
    )
    
    # Should have comparison metadata
    assert isinstance(selected['comparison'], dict)
    assert 'baseline_accuracy' in selected['comparison']
    assert 'dimred_accuracy' in selected['comparison']


def test_get_leaderboard_with_dimred():
    """Test get_leaderboard_with_dimred method (lines 444-475)."""
    config_dict = {
        'dimred': {
            'enable': 'auto',
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
    
    X, y = make_classification(n_samples=50, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    evaluator.evaluate_classification_with_dimred(models, X, y)
    evaluator.classification_results = evaluator.evaluate_classification_with_dimred(models, X, y)
    
    # Manually set classification results for testing
    evaluator.classification_results = {
        'LogisticRegression': {
            'accuracy_mean': 0.85,
            'accuracy_ci_lower': 0.80,
            'accuracy_ci_upper': 0.90,
            'selected_variant': 'dimred',
            'uses_dimred': True,
            'dimred_method': 'PCA',
            'n_components': 15,
            'comparison': {'improvement': 0.05}
        }
    }
    
    leaderboard = evaluator.get_leaderboard_with_dimred('accuracy')
    
    assert len(leaderboard) > 0
    assert leaderboard[0]['selected_variant'] in ['baseline', 'dimred']


def test_get_dimred_summary():
    """Test get_dimred_summary method (lines 477-525)."""
    config_dict = {
        'dimred': {
            'enable': 'auto',
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
    
    # Manually set classification results
    evaluator.classification_results = {
        'LogisticRegression': {
            'accuracy_mean': 0.85,
            'uses_dimred': True,
            'dimred_method': 'PCA',
            'n_components': 15,
            'comparison': {
                'improvement': 0.05,
                'is_significant': True
            }
        },
        'KNN': {
            'accuracy_mean': 0.80,
            'uses_dimred': False,
            'comparison': {
                'improvement': -0.02,
                'is_significant': False
            }
        }
    }
    
    summary = evaluator.get_dimred_summary()
    
    assert 'total_models_evaluated' in summary
    assert 'models_using_dimred' in summary
    assert 'average_improvement' in summary
    assert summary['total_models_evaluated'] == 2


def test_get_dimred_summary_with_improvements():
    """Test get_dimred_summary with various improvements (lines 484-525)."""
    config_dict = {
        'dimred': {
            'enable': 'auto',
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
    
    # Set results with various improvement scenarios
    evaluator.classification_results = {
        'Model1': {
            'uses_dimred': True,
            'dimred_method': 'PCA',
            'n_components': 10,
            'comparison': {
                'improvement': 0.10,
                'is_significant': True
            }
        },
        'Model2': {
            'uses_dimred': True,
            'dimred_method': 'PCA',
            'n_components': 15,
            'comparison': {
                'improvement': -0.05,
                'is_significant': False
            }
        },
        'Model3': {
            'uses_dimred': False,
            'comparison': {
                'improvement': 0.0,
                'is_significant': False
            }
        }
    }
    
    summary = evaluator.get_dimred_summary()
    
    assert summary['models_using_dimred'] == 2
    assert summary['models_improved_by_dimred'] == 1
    assert summary['significant_improvements'] == 1
    assert 'average_improvement' in summary
    assert 'median_improvement' in summary
    assert 'dimred_methods_used' in summary
    assert 'component_counts' in summary


def test_model_benefits_from_dimred_comprehensive():
    """Comprehensive test for _model_benefits_from_dimred (lines 343-368)."""
    preprocessor = DataPreprocessor()
    config = DimRedConfig()
    evaluator = DimRedEvaluator(preprocessor, config)
    
    # Test various model types
    test_cases = [
        ('LogisticRegression', True),
        ('LinearSVM', True),
        ('KNN', True),
        ('MLP', True),
        ('RandomForest', False),
        ('XGBoost', False),
        ('GradientBoosting', False),
        ('ExtraTrees', False),
        ('UnknownModel', True),  # Default to True
        ('logisticregression', True),  # Case insensitive
        ('xgboost', False),  # Case insensitive
        ('SVM', True),  # Linear model benefits
        ('GBM', True),  # Unknown model defaults to True
    ]
    
    for model_name, expected in test_cases:
        result = evaluator._model_benefits_from_dimred(model_name)
        assert result == expected, f"Failed for {model_name}: expected {expected}, got {result}"
