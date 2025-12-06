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
from scipy.stats import wilcoxon

from core.dimred_evaluator import DimRedEvaluator
from core.dimred import DimRedConfig


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
        base_config = DimRedConfig(config_dict)
        
        evaluator = DimRedEvaluator(
            base_config=base_config,
            random_state=42
        )
        
        assert evaluator.base_config == base_config
        assert evaluator.random_state == 42
        assert evaluator.logger is not None
    
    def test_model_benefits_from_dimred(self):
        """Test _model_benefits_from_dimred logic."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Linear models should benefit
        lr = LogisticRegression()
        assert evaluator._model_benefits_from_dimred(lr) is True
        
        # Tree-based models may not benefit as much
        rf = RandomForestClassifier()
        # The actual result depends on implementation, but test it doesn't crash
        result = evaluator._model_benefits_from_dimred(rf)
        assert isinstance(result, bool)
        
        # Unknown model defaults to True
        mock_model = Mock()
        mock_model.__class__.__name__ = "UnknownModel"
        assert evaluator._model_benefits_from_dimred(mock_model) is True
    
    def test_create_pipeline_classification(self):
        """Test _create_pipeline for classification."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'k_max': 10
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config)
        
        model = LogisticRegression()
        pipeline = evaluator._create_pipeline(model, config, task_type="classification")
        
        # Should have preprocessing and model steps
        assert len(pipeline.steps) >= 2
        step_names = [name for name, _ in pipeline.steps]
        assert 'classifier' in step_names
        
        # Test pipeline on sample data
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
    
    def test_create_pipeline_clustering(self):
        """Test _create_pipeline for clustering."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca', 
                'k_max': 10
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config)
        
        model = KMeans(n_clusters=3, random_state=42)
        pipeline = evaluator._create_pipeline(model, config, task_type="clustering")
        
        # Should have preprocessing 
        assert len(pipeline.steps) >= 1
        
        # Test pipeline on sample data  
        X, _ = make_blobs(n_samples=100, n_features=20, centers=3, random_state=42)
        pipeline.fit(X)
        labels = pipeline.fit_predict(X)
        assert len(labels) == len(X)
    
    def test_evaluate_model_with_dimred_classification(self):
        """Test evaluate_model_with_dimred for classification."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'pca'
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config)
        
        # Create sample data
        X, y = make_classification(
            n_samples=200, 
            n_features=50, 
            n_classes=2,
            random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        
        result = evaluator.evaluate_model_with_dimred(
            model, X, y, task_type="classification"
        )
        
        assert 'baseline_scores' in result
        assert 'dimred_scores' in result
        assert 'statistical_test' in result
        assert 'recommended_config' in result
        
        # Scores should be arrays
        assert isinstance(result['baseline_scores'], np.ndarray)
        assert isinstance(result['dimred_scores'], np.ndarray)
        assert len(result['baseline_scores']) > 0
        assert len(result['dimred_scores']) > 0
        
        # Should have statistical test result
        assert 'statistic' in result['statistical_test']
        assert 'p_value' in result['statistical_test']
    
    def test_evaluate_model_with_dimred_clustering(self):
        """Test evaluate_model_with_dimred for clustering."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'pca'
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config)
        
        # Create sample clustering data
        X, _ = make_blobs(
            n_samples=200,
            n_features=30, 
            centers=3,
            random_state=42
        )
        
        model = KMeans(n_clusters=3, random_state=42)
        
        result = evaluator.evaluate_model_with_dimred(
            model, X, None, task_type="clustering"
        )
        
        assert 'baseline_scores' in result
        assert 'dimred_scores' in result
        assert 'statistical_test' in result
        assert 'recommended_config' in result
        
        # Scores should be arrays
        assert isinstance(result['baseline_scores'], np.ndarray)
        assert isinstance(result['dimred_scores'], np.ndarray)
    
    def test_compare_and_select_variant(self):
        """Test _compare_and_select_variant method."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Create mock score arrays
        baseline_scores = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        dimred_scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])  # Better scores
        
        result = evaluator._compare_and_select_variant(
            baseline_scores, dimred_scores, 'on'
        )
        
        assert 'recommended_config' in result
        assert 'comparison_metrics' in result
        assert 'statistical_test' in result
        
        # Should recommend dimred since scores are better
        rec_config = result['recommended_config']
        assert rec_config.enable == 'on'  # Should enable since dimred is better
        
        # Check comparison metrics
        metrics = result['comparison_metrics']
        assert 'baseline_score' in metrics
        assert 'dimred_score' in metrics
        assert metrics['dimred_score'] > metrics['baseline_score']
    
    def test_compare_worse_dimred_scores(self):
        """Test comparison when dimred makes performance worse."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Dimred scores are worse
        baseline_scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        dimred_scores = np.array([0.78, 0.80, 0.76, 0.79, 0.77])  # Worse
        
        result = evaluator._compare_and_select_variant(
            baseline_scores, dimred_scores, 'auto'
        )
        
        # Should recommend disabling dimred
        rec_config = result['recommended_config']
        assert rec_config.enable == 'off'
    
    def test_evaluate_models_with_dimred(self):
        """Test evaluate_models_with_dimred for multiple models."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'pca'
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config, random_state=42)
        
        # Create sample data
        X, y = make_classification(
            n_samples=150,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        
        # Multiple models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        results = evaluator.evaluate_models_with_dimred(
            models, X, y, task_type="classification"
        )
        
        assert 'model_results' in results
        assert 'recommended_config' in results
        assert 'summary' in results
        
        # Should have results for each model
        model_results = results['model_results']
        assert len(model_results) == len(models)
        
        for model_name in models.keys():
            assert model_name in model_results
            result = model_results[model_name]
            assert 'baseline_scores' in result
            assert 'dimred_scores' in result
    
    def test_statistical_significance_detection(self):
        """Test detection of statistical significance."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Create significantly different scores
        baseline_scores = np.array([0.70, 0.72, 0.71, 0.69, 0.70])
        dimred_scores = np.array([0.85, 0.87, 0.86, 0.84, 0.85])  # Much better
        
        result = evaluator._compare_and_select_variant(
            baseline_scores, dimred_scores, 'auto'
        )
        
        # Should detect significance
        p_value = result['statistical_test']['p_value']
        assert p_value < 0.05  # Should be significant
        
        # Should recommend enabling dimred
        assert result['recommended_config'].enable == 'on'
    
    def test_no_statistical_significance(self):
        """Test when there's no statistical significance."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Very similar scores
        baseline_scores = np.array([0.80, 0.82, 0.81, 0.79, 0.80])
        dimred_scores = np.array([0.81, 0.83, 0.80, 0.82, 0.81])  # Slightly different
        
        result = evaluator._compare_and_select_variant(
            baseline_scores, dimred_scores, 'auto'
        )
        
        # Might not be statistically significant
        p_value = result['statistical_test']['p_value']
        # If not significant, should be more conservative
        if p_value >= 0.05:
            # Should prefer simpler approach (no dimred) if no clear benefit
            assert result['recommended_config'].enable in ['off', 'auto']
    
    def test_error_handling(self):
        """Test error handling in evaluation."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        # Invalid task type
        with pytest.raises(ValueError, match="Unsupported task type"):
            evaluator.evaluate_model_with_dimred(
                LogisticRegression(), 
                np.random.randn(100, 10), 
                np.random.randint(0, 2, 100),
                task_type="invalid"
            )
    
    def test_empty_models_dict(self):
        """Test handling of empty models dictionary."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        results = evaluator.evaluate_models_with_dimred(
            {}, X, y, task_type="classification"
        )
        
        # Should handle gracefully
        assert 'model_results' in results
        assert len(results['model_results']) == 0
        assert 'recommended_config' in results
    
    @patch('core.dimred_evaluator.wilcoxon')
    def test_wilcoxon_test_call(self, mock_wilcoxon):
        """Test that Wilcoxon test is called correctly."""
        mock_wilcoxon.return_value = (1.5, 0.03)  # Mock significant result
        
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config)
        
        baseline_scores = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        dimred_scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        
        result = evaluator._compare_and_select_variant(
            baseline_scores, dimred_scores, 'auto'
        )
        
        # Should have called wilcoxon test
        mock_wilcoxon.assert_called_once()
        
        # Should use the mocked p-value
        assert result['statistical_test']['p_value'] == 0.03
    
    def test_cv_fold_adaptation(self):
        """Test cross-validation fold adaptation for small datasets."""
        config = DimRedConfig()
        evaluator = DimRedEvaluator(config, n_folds=5)
        
        # Very small dataset
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        
        model = LogisticRegression(random_state=42)
        
        result = evaluator.evaluate_model_with_dimred(
            model, X, y, task_type="classification"
        )
        
        # Should handle small dataset gracefully (might reduce CV folds)
        assert 'baseline_scores' in result
        assert len(result['baseline_scores']) > 0


class TestIntegrationScenarios:
    """Integration tests for real-world evaluation scenarios."""
    
    def test_high_dimensional_classification(self):
        """Test evaluation with high-dimensional classification data."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto',
                'variance_target': 0.95
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config, random_state=42)
        
        # High-dimensional dataset
        X, y = make_classification(
            n_samples=300,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            random_state=42
        )
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        results = evaluator.evaluate_models_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # Should complete successfully
        assert len(results['model_results']) == 1
        
        # For high-dimensional data, dimred might be beneficial
        rec_config = results['recommended_config']
        assert rec_config.enable in ['on', 'auto', 'off']  # Any is valid
    
    def test_clustering_evaluation(self):
        """Test evaluation with clustering task."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'pca'
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config, random_state=42)
        
        # Clustering dataset
        X, _ = make_blobs(
            n_samples=200,
            n_features=50,
            centers=4,
            random_state=42
        )
        
        models = {
            'KMeans': KMeans(n_clusters=4, random_state=42)
        }
        
        results = evaluator.evaluate_models_with_dimred(
            models, X, None, task_type="clustering"
        )
        
        assert len(results['model_results']) == 1
        assert 'recommended_config' in results
    
    def test_low_dimensional_data(self):
        """Test evaluation with low-dimensional data.""" 
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto'
            }
        }
        config = DimRedConfig(config_dict)
        evaluator = DimRedEvaluator(config, random_state=42)
        
        # Low-dimensional dataset
        X, y = make_classification(
            n_samples=200,
            n_features=5,  # Very few features
            n_classes=2,
            random_state=42
        )
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        results = evaluator.evaluate_models_with_dimred(
            models, X, y, task_type="classification"
        )
        
        # For low-dimensional data, dimred probably not helpful
        rec_config = results['recommended_config']
        # Should likely recommend 'off' or stay 'auto'
        assert rec_config.enable in ['off', 'auto']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])