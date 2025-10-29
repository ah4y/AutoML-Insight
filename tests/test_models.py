"""Tests for supervised models."""

import pytest
import numpy as np
from core.models_supervised import get_supervised_models, MLPClassifier


def test_get_supervised_models():
    """Test model dictionary retrieval."""
    models = get_supervised_models(random_state=42)
    
    assert len(models) > 0
    assert 'LogisticRegression' in models
    assert 'RandomForest' in models
    assert 'XGBoost' in models
    assert 'MLP' in models


def test_mlp_classifier_training(sample_data):
    """Test MLP classifier training."""
    X, y = sample_data
    
    # Use small network for testing
    mlp = MLPClassifier(
        hidden_layers=(32,),
        max_epochs=10,
        patience=5,
        random_state=42
    )
    
    # Fit
    mlp.fit(X.values, y.values)
    
    # Predict
    predictions = mlp.predict(X.values)
    probas = mlp.predict_proba(X.values)
    
    assert predictions.shape[0] == X.shape[0]
    assert probas.shape == (X.shape[0], 3)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_models_fit_predict(sample_data):
    """Test that all models can fit and predict."""
    X, y = sample_data
    models = get_supervised_models(random_state=42)
    
    for name, model in models.items():
        # Fit
        model.fit(X.values, y.values)
        
        # Predict
        predictions = model.predict(X.values)
        
        assert predictions.shape[0] == X.shape[0]
        
        # Check if model has predict_proba
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X.values)
            assert probas.shape[0] == X.shape[0]
