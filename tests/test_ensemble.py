"""Unit tests for core/ensemble.py module.

Tests ensemble methods including weighted voting, stacking, and adaptive ensembles.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from core.ensemble import WeightedEnsemble, StackingEnsemble, AdaptiveEnsemble


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=150,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def trained_models(sample_data):
    """Create and train sample models."""
    X, y = sample_data
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(kernel='rbf', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
    }
    
    # Train all models
    for model in models.values():
        model.fit(X, y)
    
    return models, X, y


class TestWeightedEnsembleInit:
    """Test WeightedEnsemble initialization."""
    
    def test_init_with_models(self, sample_data):
        """Test initialization with models."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        
        assert ensemble.models == models
        assert ensemble.weights is None
        assert ensemble.classes_ is None
    
    def test_init_with_weights(self, sample_data):
        """Test initialization with custom weights."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        weights = [0.6, 0.4]
        
        ensemble = WeightedEnsemble(models=models, weights=weights)
        
        assert ensemble.models == models
        assert ensemble.weights == weights


class TestWeightedEnsembleFitPredict:
    """Test WeightedEnsemble fit and predict."""
    
    def test_fit(self, sample_data):
        """Test fitting the ensemble."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        ensemble.fit(X, y)
        
        assert ensemble.classes_ is not None
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        # Weights should be normalized
        assert np.isclose(sum(ensemble.weights), 1.0)
    
    def test_fit_with_custom_weights(self, sample_data):
        """Test fitting with custom weights."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        weights = [2.0, 1.0]
        
        ensemble = WeightedEnsemble(models=models, weights=weights)
        ensemble.fit(X, y)
        
        # Weights should be normalized to sum to 1
        assert np.isclose(sum(ensemble.weights), 1.0)
        # Relative weights should be preserved
        assert ensemble.weights[0] > ensemble.weights[1]
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        ensemble.fit(X, y)
        
        probas = ensemble.predict_proba(X[:10])
        
        assert probas.shape == (10, 2)  # 10 samples, 2 classes
        # Probabilities should be normalized
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_predict(self, sample_data):
        """Test class prediction."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(p in ensemble.classes_ for p in predictions)
    
    def test_predict_all_data(self, sample_data):
        """Test prediction on full dataset."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(y)
        assert all(p in ensemble.classes_ for p in predictions)


class TestStackingEnsembleInit:
    """Test StackingEnsemble initialization."""
    
    def test_init_default(self, sample_data):
        """Test default initialization."""
        models = [
            LogisticRegression(random_state=42),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=models)
        
        assert ensemble.base_models == models
        assert ensemble.meta_model is not None
        assert ensemble.use_probas is True
        assert ensemble.classes_ is None
    
    def test_init_custom_meta_model(self, sample_data):
        """Test initialization with custom meta model."""
        base_models = [
            LogisticRegression(random_state=42),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        ensemble = StackingEnsemble(
            base_models=base_models,
            meta_model=meta_model,
            use_probas=False
        )
        
        assert ensemble.base_models == base_models
        assert ensemble.meta_model is meta_model
        assert ensemble.use_probas is False


class TestStackingEnsembleFitPredict:
    """Test StackingEnsemble fit and predict."""
    
    def test_fit(self, sample_data):
        """Test fitting the stacking ensemble."""
        X, y = sample_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42),
            RandomForestClassifier(n_estimators=50, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=base_models)
        ensemble.fit(X, y)
        
        assert ensemble.classes_ is not None
        assert len(ensemble.classes_) == 2
    
    def test_meta_features_generation(self, sample_data):
        """Test meta-feature generation."""
        X, y = sample_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=base_models, use_probas=True)
        ensemble.fit(X, y)
        
        meta_features = ensemble._generate_meta_features(X[:5])
        
        # With 2 models and use_probas=True, should have 2*2=4 meta-features per sample
        assert meta_features.shape[0] == 5
        assert meta_features.shape[1] > 0
    
    def test_meta_features_no_probas(self, sample_data):
        """Test meta-feature generation without probabilities."""
        X, y = sample_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=base_models, use_probas=False)
        ensemble.fit(X, y)
        
        meta_features = ensemble._generate_meta_features(X[:5])
        
        assert meta_features.shape[0] == 5
        assert meta_features.shape[1] == len(base_models)  # One prediction per model
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=base_models)
        ensemble.fit(X, y)
        
        probas = ensemble.predict_proba(X[:10])
        
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_predict(self, sample_data):
        """Test class prediction."""
        X, y = sample_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42),
            RandomForestClassifier(n_estimators=50, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=base_models)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(p in ensemble.classes_ for p in predictions)


class TestAdaptiveEnsembleInit:
    """Test AdaptiveEnsemble initialization."""
    
    def test_init(self):
        """Test initialization."""
        ensemble = AdaptiveEnsemble(random_state=42)
        
        assert ensemble.random_state == 42
        assert ensemble.ensemble is None
        assert ensemble.ensemble_type is None
    
    def test_init_custom_seed(self):
        """Test initialization with custom seed."""
        ensemble = AdaptiveEnsemble(random_state=123)
        
        assert ensemble.random_state == 123


class TestAdaptiveEnsembleCreate:
    """Test AdaptiveEnsemble ensemble creation."""
    
    def test_create_weighted_ensemble(self, sample_data):
        """Test creating weighted ensemble from 2 models."""
        X, y = sample_data
        
        models_dict = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVC': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train models
        for model in models_dict.values():
            model.fit(X, y)
        
        evaluation_results = {
            'LogisticRegression': {
                'accuracy_mean': 0.85,
                'accuracy_scores': [0.82, 0.88, 0.85]
            },
            'SVC': {
                'accuracy_mean': 0.80,
                'accuracy_scores': [0.78, 0.82, 0.80]
            }
        }
        
        adaptive = AdaptiveEnsemble(random_state=42)
        ensemble = adaptive.create_ensemble(
            models_dict=models_dict,
            evaluation_results=evaluation_results,
            X=X,
            y=y,
            top_k=2
        )
        
        assert ensemble is not None
        assert adaptive.ensemble_type == 'weighted'
        predictions = ensemble.predict(X[:10])
        assert len(predictions) == 10
    
    def test_create_stacking_ensemble(self, sample_data):
        """Test creating stacking ensemble from 3+ models."""
        X, y = sample_data
        
        models_dict = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVC': SVC(kernel='rbf', probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        # Train models
        for model in models_dict.values():
            model.fit(X, y)
        
        evaluation_results = {
            'LogisticRegression': {
                'accuracy_mean': 0.85,
                'accuracy_scores': [0.82, 0.88, 0.85]
            },
            'SVC': {
                'accuracy_mean': 0.82,
                'accuracy_scores': [0.80, 0.84, 0.82]
            },
            'RandomForest': {
                'accuracy_mean': 0.88,
                'accuracy_scores': [0.85, 0.90, 0.88]
            }
        }
        
        adaptive = AdaptiveEnsemble(random_state=42)
        ensemble = adaptive.create_ensemble(
            models_dict=models_dict,
            evaluation_results=evaluation_results,
            X=X,
            y=y,
            top_k=3
        )
        
        assert ensemble is not None
        assert adaptive.ensemble_type == 'stacking'
        predictions = ensemble.predict(X[:10])
        assert len(predictions) == 10
    
    def test_create_with_empty_results(self, sample_data):
        """Test creation with empty evaluation results."""
        X, y = sample_data
        
        models_dict = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        models_dict['LogisticRegression'].fit(X, y)
        
        adaptive = AdaptiveEnsemble()
        ensemble = adaptive.create_ensemble(
            models_dict=models_dict,
            evaluation_results={},
            X=X,
            y=y
        )
        
        assert ensemble is None
    
    def test_top_k_selection(self, sample_data):
        """Test top-k model selection."""
        X, y = sample_data
        
        models_dict = {
            'Model1': LogisticRegression(random_state=42, max_iter=1000),
            'Model2': SVC(kernel='rbf', probability=True, random_state=42),
            'Model3': RandomForestClassifier(n_estimators=50, random_state=42),
            'Model4': KNeighborsClassifier(n_neighbors=5)
        }
        
        for model in models_dict.values():
            model.fit(X, y)
        
        evaluation_results = {
            'Model1': {'accuracy_mean': 0.90, 'accuracy_scores': [0.85, 0.95]},
            'Model2': {'accuracy_mean': 0.85, 'accuracy_scores': [0.80, 0.90]},
            'Model3': {'accuracy_mean': 0.88, 'accuracy_scores': [0.83, 0.93]},
            'Model4': {'accuracy_mean': 0.75, 'accuracy_scores': [0.70, 0.80]}
        }
        
        adaptive = AdaptiveEnsemble()
        ensemble = adaptive.create_ensemble(
            models_dict=models_dict,
            evaluation_results=evaluation_results,
            X=X,
            y=y,
            top_k=2
        )
        
        # Should select top 2 models
        assert ensemble is not None
        assert adaptive.ensemble_type == 'weighted'


class TestAdaptiveEnsembleInfo:
    """Test AdaptiveEnsemble info retrieval."""
    
    def test_get_ensemble_info(self, sample_data):
        """Test getting ensemble information."""
        X, y = sample_data
        
        models_dict = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVC': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        for model in models_dict.values():
            model.fit(X, y)
        
        evaluation_results = {
            'LogisticRegression': {
                'accuracy_mean': 0.85,
                'accuracy_scores': [0.82, 0.88, 0.85]
            },
            'SVC': {
                'accuracy_mean': 0.80,
                'accuracy_scores': [0.78, 0.82, 0.80]
            }
        }
        
        adaptive = AdaptiveEnsemble()
        adaptive.create_ensemble(
            models_dict=models_dict,
            evaluation_results=evaluation_results,
            X=X,
            y=y
        )
        
        info = adaptive.get_ensemble_info()
        
        assert info['type'] == 'weighted'
        assert info['n_models'] == 2
        assert info['weights'] is not None


class TestEnsembleScibility:
    """Test ensemble with different sizes and configurations."""
    
    def test_two_model_ensemble(self, sample_data):
        """Test ensemble with 2 models."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42)
        ]
        
        ensemble = WeightedEnsemble(models=models)
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_five_model_ensemble(self, sample_data):
        """Test ensemble with 5 models."""
        X, y = sample_data
        
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(kernel='rbf', probability=True, random_state=42),
            RandomForestClassifier(n_estimators=50, random_state=42),
            KNeighborsClassifier(n_neighbors=5),
            GradientBoostingClassifier(n_estimators=50, random_state=42)
        ]
        
        ensemble = StackingEnsemble(base_models=models)
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(y)
