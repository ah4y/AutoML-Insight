"""Unit tests for core/overfitting_detector.py module.

Tests the overfitting detection system including warning generation,
severity levels, and detection of various overfitting scenarios.
"""

import pytest
import numpy as np
from core.overfitting_detector import OverfittingDetector, OverfittingWarning


class TestOverfittingDetectorInit:
    """Test OverfittingDetector initialization."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = OverfittingDetector()
        assert detector.warnings == []
    
    def test_warning_dataclass(self):
        """Test OverfittingWarning dataclass creation."""
        warning = OverfittingWarning(
            severity='HIGH',
            warning_type='TEST_TYPE',
            message='Test message',
            recommendations=['Rec1', 'Rec2']
        )
        assert warning.severity == 'HIGH'
        assert warning.warning_type == 'TEST_TYPE'
        assert warning.message == 'Test message'
        assert len(warning.recommendations) == 2


class TestTrainTestGapDetection:
    """Test train-test gap detection."""
    
    def test_no_gap(self):
        """Test when there's no gap between train and test."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85, 'f1_macro': 0.84}
        test_scores = {'accuracy': 0.84, 'f1_macro': 0.83}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        # No overfitting warnings should be generated
        overfitting_warnings = [w for w in warnings if 'OVERFITTING' in w.warning_type]
        assert len(overfitting_warnings) == 0
    
    def test_minor_gap(self):
        """Test detection of minor overfitting (10-15% gap)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.90, 'f1_macro': 0.89}
        test_scores = {'accuracy': 0.80, 'f1_macro': 0.79}  # 10% gap
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        minor_warnings = [w for w in warnings if w.warning_type == 'MINOR_OVERFITTING']
        assert len(minor_warnings) == 1
        assert minor_warnings[0].severity == 'LOW'
    
    def test_moderate_gap(self):
        """Test detection of moderate overfitting (15-20% gap)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.95, 'f1_macro': 0.94}
        test_scores = {'accuracy': 0.78, 'f1_macro': 0.77}  # ~17% gap
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        moderate_warnings = [w for w in warnings if w.warning_type == 'MODERATE_OVERFITTING']
        assert len(moderate_warnings) == 1
        assert moderate_warnings[0].severity == 'MEDIUM'
    
    def test_severe_gap(self):
        """Test detection of severe overfitting (>20% gap)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.99, 'f1_macro': 0.98}
        test_scores = {'accuracy': 0.75, 'f1_macro': 0.74}  # ~24% gap
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        severe_warnings = [w for w in warnings if w.warning_type == 'SEVERE_OVERFITTING']
        assert len(severe_warnings) == 1
        assert severe_warnings[0].severity == 'HIGH'
        assert 'recommendations' in severe_warnings[0].__dict__


class TestPerfectScoreDetection:
    """Test perfect score detection for data leakage."""
    
    def test_perfect_score_small_data(self):
        """Test perfect score on small test set."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 1.0, 'f1_macro': 1.0}
        test_scores = {'accuracy': 0.999, 'f1_macro': 0.999}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 20}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        perfect_warnings = [w for w in warnings if 'PERFECT_SCORE' in w.warning_type]
        assert len(perfect_warnings) == 1
        assert perfect_warnings[0].severity == 'HIGH'
    
    def test_perfect_score_large_data(self):
        """Test perfect score on large test set."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 1.0, 'f1_macro': 1.0}
        test_scores = {'accuracy': 0.999, 'f1_macro': 0.999}
        cv_scores = {}
        dataset_info = {'n_samples': 10000, 'n_test_samples': 1000}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        leakage_warnings = [w for w in warnings if w.warning_type == 'PERFECT_SCORE_LEAKAGE']
        assert len(leakage_warnings) == 1
        assert leakage_warnings[0].severity == 'HIGH'
    
    def test_very_high_score_small_test(self):
        """Test high score on small test set."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.99, 'f1_macro': 0.99}
        test_scores = {'accuracy': 0.99, 'f1_macro': 0.99}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 10}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        high_score_warnings = [w for w in warnings if w.warning_type == 'VERY_HIGH_SCORE_SMALL']
        assert len(high_score_warnings) == 1
        assert high_score_warnings[0].severity == 'MEDIUM'
    
    def test_normal_good_score(self):
        """Test normal good score (no warning)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.92, 'f1_macro': 0.91}
        test_scores = {'accuracy': 0.90, 'f1_macro': 0.89}
        cv_scores = {}
        dataset_info = {'n_samples': 1000, 'n_test_samples': 250}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        perfect_warnings = [w for w in warnings if 'PERFECT' in w.warning_type]
        assert len(perfect_warnings) == 0


class TestCVVarianceDetection:
    """Test cross-validation variance detection."""
    
    def test_normal_cv_variance(self):
        """Test normal CV variance (no warning)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85}
        test_scores = {'accuracy': 0.83}
        cv_scores = {'accuracy': [0.82, 0.83, 0.84, 0.85, 0.86]}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        cv_warnings = [w for w in warnings if w.warning_type == 'LOW_CV_VARIANCE']
        assert len(cv_warnings) == 0
    
    def test_suspiciously_consistent_cv(self):
        """Test suspiciously consistent CV scores."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.95}
        test_scores = {'accuracy': 0.94}
        # Very low variance on high scores
        cv_scores = {'accuracy': [0.950, 0.951, 0.950, 0.952, 0.951]}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        cv_warnings = [w for w in warnings if w.warning_type == 'LOW_CV_VARIANCE']
        assert len(cv_warnings) == 1
        assert cv_warnings[0].severity == 'MEDIUM'
    
    def test_low_variance_low_scores(self):
        """Test low variance with low scores (no warning)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.50}
        test_scores = {'accuracy': 0.48}
        cv_scores = {'accuracy': [0.475, 0.485, 0.480, 0.490, 0.485]}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        cv_warnings = [w for w in warnings if w.warning_type == 'LOW_CV_VARIANCE']
        assert len(cv_warnings) == 0


class TestTestSetSizeDetection:
    """Test test set size detection."""
    
    def test_very_small_test_set(self):
        """Test detection of very small test set."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85}
        test_scores = {'accuracy': 0.83}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 20}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        size_warnings = [w for w in warnings if w.warning_type == 'SMALL_TEST_SET']
        assert len(size_warnings) == 1
        assert size_warnings[0].severity == 'HIGH'
    
    def test_imbalanced_test_set(self):
        """Test detection of few samples per class."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85}
        test_scores = {'accuracy': 0.83}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 50, 'n_classes': 10}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        imbalance_warnings = [w for w in warnings if w.warning_type == 'FEW_SAMPLES_PER_CLASS']
        assert len(imbalance_warnings) == 1
        assert imbalance_warnings[0].severity == 'MEDIUM'
    
    def test_adequate_test_set(self):
        """Test adequate test set (no warning)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85}
        test_scores = {'accuracy': 0.83}
        cv_scores = {}
        dataset_info = {'n_samples': 1000, 'n_test_samples': 250, 'n_classes': 2}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        size_warnings = [w for w in warnings if 'TEST_SET' in w.warning_type]
        assert len(size_warnings) == 0


class TestImbalancedDataDetection:
    """Test imbalanced data detection."""
    
    def test_majority_class_bias(self):
        """Test detection of majority class bias."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.92, 'f1_macro': 0.55}
        test_scores = {'accuracy': 0.92, 'f1_macro': 0.55}
        cv_scores = {}
        dataset_info = {
            'n_samples': 100,
            'n_test_samples': 25,
            'class_balance': {0: 90, 1: 10}  # 9:1 imbalance
        }
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        bias_warnings = [w for w in warnings if w.warning_type == 'MAJORITY_CLASS_BIAS']
        assert len(bias_warnings) == 1
        assert bias_warnings[0].severity == 'HIGH'
    
    def test_balanced_high_performance(self):
        """Test high performance on imbalanced data (legitimate)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.96, 'f1_macro': 0.95}
        test_scores = {'accuracy': 0.96, 'f1_macro': 0.95}
        cv_scores = {}
        dataset_info = {
            'n_samples': 100,
            'n_test_samples': 25,
            'class_balance': {0: 90, 1: 10}
        }
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        bias_warnings = [w for w in warnings if w.warning_type == 'MAJORITY_CLASS_BIAS']
        assert len(bias_warnings) == 0
    
    def test_low_imbalance_ratio(self):
        """Test low imbalance ratio (no bias detection)."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.90, 'f1_macro': 0.50}
        test_scores = {'accuracy': 0.90, 'f1_macro': 0.50}
        cv_scores = {}
        dataset_info = {
            'n_samples': 100,
            'n_test_samples': 25,
            'class_balance': {0: 60, 1: 40}  # 1.5:1 imbalance
        }
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        bias_warnings = [w for w in warnings if w.warning_type == 'MAJORITY_CLASS_BIAS']
        assert len(bias_warnings) == 0


class TestMultipleWarnings:
    """Test detection of multiple issues."""
    
    def test_multiple_warnings(self):
        """Test that multiple issues generate multiple warnings."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.99, 'f1_macro': 0.98}
        test_scores = {'accuracy': 0.75, 'f1_macro': 0.75}  # Large gap
        cv_scores = {'accuracy': [0.755, 0.751, 0.749]}  # Low variance
        dataset_info = {
            'n_samples': 100,
            'n_test_samples': 20,  # Small test set
            'n_classes': 2,
            'class_balance': {0: 95, 1: 5}  # Imbalanced
        }
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        # Should have multiple warnings
        assert len(warnings) >= 3
        
        # Check for expected warning types
        warning_types = {w.warning_type for w in warnings}
        assert 'SEVERE_OVERFITTING' in warning_types or 'MODERATE_OVERFITTING' in warning_types
        assert 'SMALL_TEST_SET' in warning_types


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_warnings(self):
        """Test detection with minimal data."""
        detector = OverfittingDetector()
        warnings = detector.detect_overfitting({}, {}, {}, {})
        assert isinstance(warnings, list)
    
    def test_missing_metrics(self):
        """Test handling of missing metrics."""
        detector = OverfittingDetector()
        train_scores = {'precision': 0.85}
        test_scores = {'precision': 0.83}
        cv_scores = {}
        dataset_info = {'n_samples': 100}
        
        # Should not crash with missing accuracy/f1_macro
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        assert isinstance(warnings, list)
    
    def test_extreme_scores(self):
        """Test extreme score values."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.0, 'f1_macro': 0.0}
        test_scores = {'accuracy': 0.0, 'f1_macro': 0.0}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 25}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        assert isinstance(warnings, list)
    
    def test_large_cv_scores_list(self):
        """Test with many CV folds."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.85}
        test_scores = {'accuracy': 0.83}
        cv_scores = {'accuracy': np.random.uniform(0.80, 0.85, 50).tolist()}
        dataset_info = {'n_samples': 100}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        assert isinstance(warnings, list)
    
    def test_recommendation_presence(self):
        """Test that all warnings include recommendations."""
        detector = OverfittingDetector()
        train_scores = {'accuracy': 0.99, 'f1_macro': 0.98}
        test_scores = {'accuracy': 0.70, 'f1_macro': 0.70}
        cv_scores = {}
        dataset_info = {'n_samples': 100, 'n_test_samples': 20}
        
        warnings = detector.detect_overfitting(train_scores, test_scores, cv_scores, dataset_info)
        
        for warning in warnings:
            assert hasattr(warning, 'recommendations')
            assert isinstance(warning.recommendations, list)
            assert len(warning.recommendations) > 0
