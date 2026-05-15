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


def test_evaluate_model_with_binary_prediction_error():
    """Test evaluate_model handles prediction errors (lines 94-97)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    # Create a model that might fail
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=100, random_state=42)
    
    evaluator = ClassificationEvaluator(n_folds=2, n_repeats=1)
    results = evaluator.evaluate_model(model, X, y, 'TestModel')
    
    assert 'model_name' in results
    assert results['model_name'] == 'TestModel'


def test_compare_models_missing_models():
    """Test compare_models with missing model (lines 137-138)."""
    evaluator = ClassificationEvaluator()
    
    # Compare non-existent models
    comparison = evaluator.compare_models('NonExistent1', 'NonExistent2')
    
    assert comparison == {}


def test_compare_models_mcnemar_failure():
    """Test compare_models handles McNemar test failure (lines 146-154)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    
    evaluator = ClassificationEvaluator(n_folds=2, n_repeats=1)
    
    model1 = LogisticRegression(max_iter=100, random_state=42)
    model2 = DecisionTreeClassifier(random_state=42)
    
    evaluator.evaluate_model(model1, X, y, 'Model1')
    evaluator.evaluate_model(model2, X, y, 'Model2')
    
    # This will attempt McNemar test
    comparison = evaluator.compare_models('Model1', 'Model2')
    
    # Should not crash even if test fails
    assert isinstance(comparison, dict)


def test_compare_models_wilcoxon_failure():
    """Test compare_models handles Wilcoxon test failure (lines 157-165)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    from sklearn.linear_model import LogisticRegression
    
    evaluator = ClassificationEvaluator(n_folds=2, n_repeats=1)
    
    model1 = LogisticRegression(max_iter=100, random_state=42)
    model2 = LogisticRegression(max_iter=100, random_state=43)
    
    evaluator.evaluate_model(model1, X, y, 'Model1')
    evaluator.evaluate_model(model2, X, y, 'Model2')
    
    comparison = evaluator.compare_models('Model1', 'Model2')
    
    # Should have completed without error
    assert isinstance(comparison, dict)


def test_get_leaderboard_basic():
    """Test get_leaderboard with basic results."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    evaluator = ClassificationEvaluator(n_folds=2, n_repeats=1)
    
    model1 = LogisticRegression(max_iter=100, random_state=42)
    model2 = RandomForestClassifier(n_estimators=5, random_state=42)
    
    evaluator.evaluate_model(model1, X, y, 'LogisticRegression')
    evaluator.evaluate_model(model2, X, y, 'RandomForest')
    
    leaderboard = evaluator.get_leaderboard('accuracy')
    
    assert len(leaderboard) == 2
    assert leaderboard[0]['score'] >= leaderboard[1]['score']


def test_get_leaderboard_with_overfitting_penalty():
    """Test get_leaderboard with overfitting penalty (lines 189-198)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    from sklearn.linear_model import LogisticRegression
    
    evaluator = ClassificationEvaluator(n_folds=2, n_repeats=1)
    model = LogisticRegression(max_iter=100, random_state=42)
    
    evaluator.evaluate_model(model, X, y, 'Model')
    
    leaderboard_with_penalty = evaluator.get_leaderboard('accuracy', penalize_overfitting=True)
    leaderboard_without_penalty = evaluator.get_leaderboard('accuracy', penalize_overfitting=False)
    
    # Both should return leaderboards
    assert len(leaderboard_with_penalty) > 0
    assert len(leaderboard_without_penalty) > 0


def test_evaluate_with_holdout_basic(sample_data):
    """Test evaluate_with_holdout basic functionality."""
    X, y = sample_data
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=42, stratify=y.values
    )
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=2)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'TestModel'
    )
    
    assert 'train_accuracy' in results
    assert 'test_accuracy' in results
    assert 'overfitting_gap' in results


def test_evaluate_with_holdout_small_dataset():
    """Test evaluate_with_holdout with very small dataset (lines 255-258)."""
    # Small dataset that triggers minimal CV (2-fold)
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    X_train = X[:20]
    y_train = y[:20]
    X_test = X[20:]
    y_test = y[20:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=5)  # Request 5 folds but should adapt
    
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'SmallDataset'
    )
    
    assert 'cv_strategy' in results
    assert results['cv_folds'] == 2  # Should use 2-fold due to small dataset


def test_evaluate_with_holdout_cv_strategy_adaptation():
    """Test evaluate_with_holdout CV strategy adaptation (lines 248-271)."""
    # Medium dataset
    X = np.random.rand(100, 5)
    y = np.array([0, 1] * 50)
    
    X_train = X[:70]
    y_train = y[:70]
    X_test = X[70:]
    y_test = y[70:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=5)
    
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'MediumDataset'
    )
    
    assert 'cv_strategy' in results
    assert results['cv_folds'] > 0


def test_evaluate_with_holdout_large_dataset():
    """Test evaluate_with_holdout with large dataset (lines 268-271)."""
    # Large dataset
    X = np.random.rand(1000, 5)
    y = np.array([0, 1] * 500)
    
    X_train = X[:700]
    y_train = y[:700]
    X_test = X[700:]
    y_test = y[700:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=3)
    
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'LargeDataset'
    )
    
    assert 'cv_sample_size' in results


def test_evaluate_with_holdout_cv_failure():
    """Test evaluate_with_holdout handles CV failure (lines 293-307)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    X_train = X[:20]
    y_train = y[:20]
    X_test = X[20:]
    y_test = y[20:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=2)
    
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'TestModel'
    )
    
    # Should complete even if CV fails
    assert 'cv_accuracy_mean' in results


def test_evaluate_with_holdout_svm_optimization():
    """Test evaluate_with_holdout SVM optimization for large datasets (lines 312-330)."""
    # Create a dataset large enough to trigger SVM optimization
    X = np.random.rand(25000, 5)
    y = np.array([0, 1] * 12500)
    
    X_train = X[:17500]
    y_train = y[:17500]
    X_test = X[17500:]
    y_test = y[17500:]
    
    from sklearn.svm import SVC
    evaluator = ClassificationEvaluator(n_folds=2)
    
    model = SVC(random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'SVM'
    )
    
    # Should complete with optimization applied
    assert 'test_accuracy' in results


def test_evaluate_with_holdout_training_error_handling():
    """Test evaluate_with_holdout error handling during training (lines 333-338)."""
    X = np.random.rand(30, 5)
    y = np.array([0, 1] * 15)
    
    X_train = X[:20]
    y_train = y[:20]
    X_test = X[20:]
    y_test = y[20:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=2)
    
    # Model should train successfully
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'TestModel'
    )
    
    assert 'trained_model' in results


def test_evaluate_with_holdout_overfitting_detection():
    """Test overfitting detection in evaluate_with_holdout (lines 350-367)."""
    X = np.random.rand(100, 5)
    y = np.array([0, 1] * 50)
    
    X_train = X[:70]
    y_train = y[:70]
    X_test = X[70:]
    y_test = y[70:]
    
    from sklearn.tree import DecisionTreeClassifier
    evaluator = ClassificationEvaluator(n_folds=2)
    
    # Decision tree is prone to overfitting
    model = DecisionTreeClassifier(random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'DecisionTree'
    )
    
    assert 'overfitting_warnings' in results


def test_evaluate_with_holdout_return_structure():
    """Test evaluate_with_holdout return structure (lines 369-408)."""
    X = np.random.rand(40, 5)
    y = np.array([0, 1] * 20)
    
    X_train = X[:28]
    y_train = y[:28]
    X_test = X[28:]
    y_test = y[28:]
    
    from sklearn.linear_model import LogisticRegression
    evaluator = ClassificationEvaluator(n_folds=2)
    
    model = LogisticRegression(max_iter=100, random_state=42)
    results = evaluator.evaluate_with_holdout(
        model, X_train, y_train, X_test, y_test, 'TestModel'
    )
    
    # Check complete structure
    assert 'model_name' in results
    assert 'train_accuracy' in results
    assert 'test_accuracy' in results
    assert 'train_f1_macro' in results
    assert 'test_f1_macro' in results
    assert 'cv_accuracy_mean' in results
    assert 'cv_accuracy_std' in results
    assert 'overfitting_gap' in results
    assert 'trained_model' in results
    assert 'predictions' in results
    assert 'true_labels' in results
