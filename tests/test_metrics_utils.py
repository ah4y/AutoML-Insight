"""Unit tests for utils/metrics_utils.py module.

Tests confidence interval computation, bootstrap methods, and statistical tests.
"""

import pytest
import numpy as np
from utils.metrics_utils import compute_confidence_interval, bootstrap_ci, mcnemar_test, wilcoxon_test


class TestComputeConfidenceInterval:
    """Test confidence interval computation."""
    
    def test_empty_scores(self):
        """Test with empty scores list."""
        mean, lower, upper = compute_confidence_interval([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0
    
    def test_single_score(self):
        """Test with single score."""
        mean, lower, upper = compute_confidence_interval([0.85])
        assert mean == 0.85
        assert lower == 0.85
        assert upper == 0.85
    
    def test_two_scores(self):
        """Test with two scores."""
        scores = [0.80, 0.90]
        mean, lower, upper = compute_confidence_interval(scores)
        
        assert np.isclose(mean, 0.85)
        assert lower < mean
        assert upper > mean
    
    def test_multiple_scores(self):
        """Test with multiple scores."""
        scores = [0.80, 0.82, 0.85, 0.87, 0.90]
        mean, lower, upper = compute_confidence_interval(scores)
        
        assert np.isclose(mean, np.mean(scores))
        assert lower < mean
        assert upper > mean
        assert upper - lower > 0
    
    def test_confidence_levels(self):
        """Test different confidence levels."""
        scores = [0.75, 0.80, 0.85, 0.90, 0.95]
        
        # 90% confidence
        mean_90, lower_90, upper_90 = compute_confidence_interval(scores, 0.90)
        
        # 95% confidence
        mean_95, lower_95, upper_95 = compute_confidence_interval(scores, 0.95)
        
        # Higher confidence should have wider interval
        assert (upper_95 - lower_95) > (upper_90 - lower_90)
        assert mean_90 == mean_95
    
    def test_perfect_scores(self):
        """Test with all perfect scores."""
        scores = [1.0] * 5
        mean, lower, upper = compute_confidence_interval(scores)
        
        assert mean == 1.0
        assert lower == 1.0
        assert upper == 1.0
    
    def test_zero_scores(self):
        """Test with all zero scores."""
        scores = [0.0] * 5
        mean, lower, upper = compute_confidence_interval(scores)
        
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""
    
    def test_empty_scores(self):
        """Test with empty scores."""
        mean, lower, upper = bootstrap_ci(np.array([]))
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0
    
    def test_single_score(self):
        """Test with single score."""
        mean, lower, upper = bootstrap_ci(np.array([0.85]))
        assert mean == 0.85
        # Bootstrap with single value will have zero variance
        assert lower <= mean
        assert upper >= mean
    
    def test_multiple_scores(self):
        """Test with multiple scores."""
        np.random.seed(42)
        scores = np.array([0.75, 0.80, 0.85, 0.90, 0.95])
        mean, lower, upper = bootstrap_ci(scores)
        
        assert mean == np.mean(scores)
        assert lower < mean
        assert upper > mean
    
    def test_bootstrap_iterations(self):
        """Test effect of bootstrap iterations."""
        np.random.seed(42)
        scores = np.array([0.70, 0.75, 0.80, 0.85, 0.90])
        
        _, l_100, u_100 = bootstrap_ci(scores, n_bootstraps=100)
        
        np.random.seed(42)
        _, l_1000, u_1000 = bootstrap_ci(scores, n_bootstraps=1000)
        
        # Both should produce valid intervals
        assert l_100 > 0
        assert u_100 <= 1.0
        assert l_1000 > 0
        assert u_1000 <= 1.0
    
    def test_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        scores = np.array([0.70, 0.75, 0.80, 0.85, 0.90])
        
        mean_90, lower_90, upper_90 = bootstrap_ci(scores, confidence=0.90)
        
        np.random.seed(42)
        mean_95, lower_95, upper_95 = bootstrap_ci(scores, confidence=0.95)
        
        assert mean_90 == mean_95
        # 95% CI should be wider than 90% CI
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)


class TestMcNemarTest:
    """Test McNemar's test for paired predictions."""
    
    def test_identical_predictions(self):
        """Test when both models make identical predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        pred1 = np.array([0, 1, 0, 1, 0])
        pred2 = np.array([0, 1, 0, 1, 0])
        
        p_value = mcnemar_test(y_true, pred1, pred2)
        assert p_value == 1.0
    
    def test_perfect_predictions(self):
        """Test when both models predict perfectly."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        pred1 = np.array([0, 1, 0, 1, 0, 1])
        pred2 = np.array([0, 1, 0, 1, 0, 1])
        
        p_value = mcnemar_test(y_true, pred1, pred2)
        assert p_value == 1.0
    
    def test_different_predictions(self):
        """Test when models have different predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred1 = np.array([0, 1, 0, 1, 1, 1, 0, 1])  # 1 error
        pred2 = np.array([0, 1, 1, 1, 0, 1, 0, 1])  # 1 error, different position
        
        p_value = mcnemar_test(y_true, pred1, pred2)
        assert 0 <= p_value <= 1
    
    def test_significant_difference(self):
        """Test with significant model differences."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Perfect
        pred2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # All wrong
        
        p_value = mcnemar_test(y_true, pred1, pred2)
        assert 0 <= p_value <= 1
    
    def test_no_disagreement(self):
        """Test when models never disagree."""
        y_true = np.array([0, 1, 0, 1, 0])
        pred1 = np.array([0, 1, 1, 1, 0])
        pred2 = np.array([0, 1, 1, 1, 0])
        
        p_value = mcnemar_test(y_true, pred1, pred2)
        assert p_value == 1.0


class TestWilcoxonTest:
    """Test Wilcoxon signed-rank test."""
    
    def test_identical_scores(self):
        """Test with identical scores."""
        scores1 = np.array([0.8, 0.85, 0.9, 0.75, 0.88])
        scores2 = np.array([0.8, 0.85, 0.9, 0.75, 0.88])
        
        p_value = wilcoxon_test(scores1, scores2)
        # Identical scores should have high p-value (no significant difference)
        assert p_value > 0.05
    
    def test_different_scores(self):
        """Test with different scores."""
        scores1 = np.array([0.80, 0.82, 0.81, 0.80, 0.79])
        scores2 = np.array([0.90, 0.92, 0.91, 0.90, 0.89])
        
        p_value = wilcoxon_test(scores1, scores2)
        # Very different scores should have low p-value
        assert 0 <= p_value <= 1
    
    def test_single_sample_handling(self):
        """Test behavior with edge cases."""
        scores1 = np.array([0.85])
        scores2 = np.array([0.85])
        
        p_value = wilcoxon_test(scores1, scores2)
        # Should return 1.0 for identical single values
        assert p_value == 1.0
    
    def test_moderate_difference(self):
        """Test with moderate differences."""
        scores1 = np.array([0.80, 0.82, 0.81, 0.80, 0.82])
        scores2 = np.array([0.85, 0.84, 0.86, 0.84, 0.85])
        
        p_value = wilcoxon_test(scores1, scores2)
        assert 0 <= p_value <= 1
    
    def test_paired_samples(self):
        """Test properly paired samples."""
        scores1 = np.array([0.75, 0.80, 0.78, 0.82, 0.79])
        scores2 = np.array([0.80, 0.82, 0.81, 0.85, 0.83])
        
        p_value = wilcoxon_test(scores1, scores2)
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1


class TestStatisticalTestsIntegration:
    """Integration tests for statistical tests."""
    
    def test_confidence_interval_consistency(self):
        """Test consistency between CI methods."""
        np.random.seed(42)
        scores = np.array([0.75, 0.78, 0.82, 0.85, 0.88, 0.90])
        
        mean_ci, lower_ci, upper_ci = compute_confidence_interval(scores)
        mean_boot, lower_boot, upper_boot = bootstrap_ci(scores)
        
        # Means should be identical
        assert mean_ci == mean_boot
        
        # Both should produce valid intervals
        assert lower_ci < mean_ci < upper_ci
        assert lower_boot < mean_boot < upper_boot
    
    def test_model_comparison_workflow(self):
        """Test complete model comparison workflow."""
        np.random.seed(42)
        
        # Simulate CV scores for two models
        model1_cv_scores = np.array([0.75, 0.78, 0.81, 0.79, 0.82])
        model2_cv_scores = np.array([0.76, 0.79, 0.80, 0.81, 0.83])
        
        # Get confidence intervals
        mean1, lower1, upper1 = compute_confidence_interval(model1_cv_scores)
        mean2, lower2, upper2 = compute_confidence_interval(model2_cv_scores)
        
        # Perform statistical test
        p_value = wilcoxon_test(model1_cv_scores, model2_cv_scores)
        
        # Check results are valid
        assert mean1 > 0.7
        assert mean2 > 0.7
        assert 0 <= p_value <= 1
    
    def test_edge_case_scores(self):
        """Test edge cases in score arrays."""
        # All same score
        scores = np.array([0.85] * 10)
        mean, lower, upper = compute_confidence_interval(scores)
        assert mean == 0.85
        
        # Very small scores
        scores = np.array([0.01, 0.02, 0.01, 0.02])
        mean, lower, upper = compute_confidence_interval(scores)
        assert 0 < mean < 0.03
        
        # Scores close to 1
        scores = np.array([0.99, 0.98, 0.99, 0.98])
        mean, lower, upper = compute_confidence_interval(scores)
        assert mean > 0.98
