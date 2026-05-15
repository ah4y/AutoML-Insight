"""Unit tests for core/tuning.py module.

Tests the Optuna hyperparameter tuning system including optimization,
parameter space generation, and model-specific tuning strategies.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from core.tuning import OptunaHyperparameterTuner


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    return X, y


class TestOptunaHyperparameterTunerInit:
    """Test OptunaHyperparameterTuner initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        tuner = OptunaHyperparameterTuner()
        
        assert tuner.n_trials == 20
        assert tuner.cv == 3
        assert tuner.scoring == 'accuracy'
        assert tuner.random_state == 42
        assert tuner.verbose is False
        assert tuner.best_params == {}
        assert tuner.best_score is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        tuner = OptunaHyperparameterTuner(
            n_trials=50,
            cv=5,
            scoring='f1',
            random_state=123,
            verbose=True
        )
        
        assert tuner.n_trials == 50
        assert tuner.cv == 5
        assert tuner.scoring == 'f1'
        assert tuner.random_state == 123
        assert tuner.verbose is True
    
    def test_initialization_edge_cases(self):
        """Test initialization with edge case values."""
        tuner = OptunaHyperparameterTuner(
            n_trials=1,
            cv=2,
            random_state=0
        )
        
        assert tuner.n_trials == 1
        assert tuner.cv == 2


class TestParameterSpaceGeneration:
    """Test parameter space generation for different models."""
    
    def test_random_forest_param_space(self):
        """Test RandomForest parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        # Mock trial
        class MockTrial:
            def suggest_int(self, name, low, high):
                return (low + high) // 2
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'RandomForest')
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert 'min_samples_leaf' in params
        
        # Check parameter values are within expected ranges
        assert 50 <= params['n_estimators'] <= 300
        assert 3 <= params['max_depth'] <= 20
        assert 2 <= params['min_samples_split'] <= 10
        assert 1 <= params['min_samples_leaf'] <= 5
    
    def test_xgboost_param_space(self):
        """Test XGBoost parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            def suggest_int(self, name, low, high):
                return (low + high) // 2
            
            def suggest_float(self, name, low, high, log=False):
                if log:
                    return 10 ** ((np.log10(low) + np.log10(high)) / 2)
                return (low + high) / 2
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'XGBoost')
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'subsample' in params
        assert 'colsample_bytree' in params
    
    def test_rbf_svm_param_space(self):
        """Test RBF-SVM parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            def suggest_float(self, name, low, high, log=False):
                if log:
                    return 10 ** ((np.log10(low) + np.log10(high)) / 2)
                return (low + high) / 2
            
            def suggest_categorical(self, name, choices):
                return choices[0]
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'RBF-SVM')
        
        assert 'C' in params
        assert 'gamma' in params
        assert params['gamma'] in ['scale', 'auto']
    
    def test_knn_param_space(self):
        """Test KNN parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            def suggest_int(self, name, low, high):
                return (low + high) // 2
            
            def suggest_categorical(self, name, choices):
                return choices[0]
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'KNN')
        
        assert 'n_neighbors' in params
        assert 'weights' in params
        assert 'metric' in params
        assert params['weights'] in ['uniform', 'distance']
    
    def test_mlp_param_space(self):
        """Test MLP parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            def suggest_categorical(self, name, choices):
                return choices[0]
            
            def suggest_float(self, name, low, high, log=False):
                if log:
                    return 10 ** ((np.log10(low) + np.log10(high)) / 2)
                return (low + high) / 2
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'MLP')
        
        assert 'hidden_layers' in params
        assert 'dropout' in params
        assert 'learning_rate' in params
        assert isinstance(params['hidden_layers'], tuple)
    
    def test_logistic_regression_param_space(self):
        """Test LogisticRegression parameter space generation."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            def suggest_float(self, name, low, high, log=False):
                if log:
                    return 10 ** ((np.log10(low) + np.log10(high)) / 2)
                return (low + high) / 2
            
            def suggest_categorical(self, name, choices):
                return choices[0]
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'LogisticRegression')
        
        assert 'C' in params
        assert 'penalty' in params
        assert params['penalty'] == 'l2'
    
    def test_unknown_model_param_space(self):
        """Test parameter space for unknown model."""
        tuner = OptunaHyperparameterTuner(n_trials=3)
        
        class MockTrial:
            pass
        
        trial = MockTrial()
        params = tuner._get_param_space(trial, 'UnknownModel')
        
        assert params == {}


class TestTuneMethod:
    """Test the tune method."""
    
    def test_tune_logistic_regression(self, sample_data):
        """Test tuning LogisticRegression."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=3,
            cv=2,
            random_state=42,
            verbose=False
        )
        
        model = LogisticRegression(random_state=42)
        tuned_model = tuner.tune('LogisticRegression', model, X, y)
        
        # Check that tuning happened
        assert tuner.best_params is not None
        assert isinstance(tuner.best_params, dict)
        assert tuner.best_score is not None
        assert 0 <= tuner.best_score <= 1
        
        # Tuned model should be able to predict
        predictions = tuned_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_tune_random_forest(self, sample_data):
        """Test tuning RandomForest."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=3,
            cv=2,
            random_state=42,
            verbose=False
        )
        
        model = RandomForestClassifier(random_state=42)
        tuned_model = tuner.tune('RandomForest', model, X, y)
        
        assert tuner.best_params is not None
        assert tuner.best_score is not None
        
        predictions = tuned_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_tune_preserves_seed(self, sample_data):
        """Test that tuning preserves reproducibility with seed."""
        X, y = sample_data
        
        tuner1 = OptunaHyperparameterTuner(
            n_trials=2,
            cv=2,
            random_state=42,
            verbose=False
        )
        
        tuner2 = OptunaHyperparameterTuner(
            n_trials=2,
            cv=2,
            random_state=42,
            verbose=False
        )
        
        model1 = LogisticRegression(random_state=42)
        model2 = LogisticRegression(random_state=42)
        
        tuner1.tune('LogisticRegression', model1, X, y)
        tuner2.tune('LogisticRegression', model2, X, y)
        
        # With same seed, should get same best score (within floating point tolerance)
        assert np.isclose(tuner1.best_score, tuner2.best_score, atol=0.01)


class TestBestParamsTracking:
    """Test tracking of best parameters and scores."""
    
    def test_best_params_updated(self, sample_data):
        """Test that best_params are updated after tuning."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=3,
            cv=2,
            random_state=42
        )
        
        assert tuner.best_params == {}
        assert tuner.best_score is None
        
        model = LogisticRegression(random_state=42)
        tuner.tune('LogisticRegression', model, X, y)
        
        assert len(tuner.best_params) > 0
        assert tuner.best_score is not None
    
    def test_best_score_range(self, sample_data):
        """Test that best_score is in valid range."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=5,
            cv=2,
            scoring='accuracy',
            random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        tuner.tune('LogisticRegression', model, X, y)
        
        assert 0 <= tuner.best_score <= 1


class TestMultipleModels:
    """Test tuning of multiple different models."""
    
    def test_sequential_tuning(self, sample_data):
        """Test tuning multiple models sequentially."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=2,
            cv=2,
            random_state=42
        )
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            tuned = tuner.tune(name, model, X, y)
            results[name] = tuned
            
            # Each should produce a valid model
            predictions = tuned.predict(X)
            assert len(predictions) == len(y)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_trial(self, sample_data):
        """Test with single trial."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=1,
            cv=2,
            random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        tuned_model = tuner.tune('LogisticRegression', model, X, y)
        
        assert tuner.best_score is not None
        predictions = tuned_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_many_cv_folds(self, sample_data):
        """Test with many CV folds."""
        X, y = sample_data
        
        tuner = OptunaHyperparameterTuner(
            n_trials=2,
            cv=5,
            random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        tuned_model = tuner.tune('LogisticRegression', model, X, y)
        
        predictions = tuned_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_different_scoring_metrics(self, sample_data):
        """Test with different scoring metrics."""
        X, y = sample_data
        
        for scoring_metric in ['accuracy', 'f1']:
            tuner = OptunaHyperparameterTuner(
                n_trials=2,
                cv=2,
                scoring=scoring_metric,
                random_state=42
            )
            
            model = LogisticRegression(random_state=42)
            tuned_model = tuner.tune('LogisticRegression', model, X, y)
            
            assert tuner.best_score is not None
            predictions = tuned_model.predict(X)
            assert len(predictions) == len(y)
