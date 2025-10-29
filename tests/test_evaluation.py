"""Tests for evaluation."""

import pytest
import numpy as np
from core.evaluate_cls import ClassificationEvaluator
from core.models_supervised import get_supervised_models


def test_classification_evaluator(sample_data):
    """Test classification evaluation."""
    X, y = sample_data
    
    # Get a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Evaluate
    evaluator = ClassificationEvaluator(n_folds=3, n_repeats=2, random_state=42)
    results = evaluator.evaluate_model(model, X.values, y.values, 'LogisticRegression')
    
    # Check results
    assert 'accuracy_mean' in results
    assert 'f1_macro_mean' in results
    assert results['accuracy_mean'] > 0
    assert results['accuracy_mean'] <= 1.0


def test_evaluator_leaderboard(sample_data):
    """Test leaderboard generation."""
    X, y = sample_data
    
    evaluator = ClassificationEvaluator(n_folds=3, n_repeats=1, random_state=42)
    
    # Evaluate multiple models
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    for name, model in models.items():
        evaluator.evaluate_model(model, X.values, y.values, name)
    
    # Get leaderboard
    leaderboard = evaluator.get_leaderboard('accuracy')
    
    assert len(leaderboard) == 2
    assert leaderboard[0]['score'] >= leaderboard[1]['score']


def test_model_comparison(sample_data):
    """Test statistical comparison between models."""
    X, y = sample_data
    
    evaluator = ClassificationEvaluator(n_folds=3, n_repeats=2, random_state=42)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    
    model1 = LogisticRegression(max_iter=1000, random_state=42)
    model2 = DecisionTreeClassifier(random_state=42)
    
    evaluator.evaluate_model(model1, X.values, y.values, 'Model1')
    evaluator.evaluate_model(model2, X.values, y.values, 'Model2')
    
    # Compare
    comparison = evaluator.compare_models('Model1', 'Model2')
    
    # Should have p-values
    assert 'wilcoxon_p_value' in comparison or 'mcnemar_p_value' in comparison
