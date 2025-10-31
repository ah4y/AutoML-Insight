"""
Overfitting Detection and User Guidance System.
Detects unrealistic model performance and provides actionable recommendations.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class OverfittingWarning:
    """Container for overfitting warning details."""
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    warning_type: str
    message: str
    recommendations: List[str]


class OverfittingDetector:
    """
    Detects overfitting and data leakage issues.
    Provides user-friendly guidance for resolving issues.
    """
    
    def __init__(self):
        self.warnings = []
    
    def detect_overfitting(
        self,
        train_scores: Dict[str, float],
        test_scores: Dict[str, float],
        cv_scores: Dict[str, List[float]],
        dataset_info: Dict[str, Any]
    ) -> List[OverfittingWarning]:
        """
        Comprehensive overfitting detection.
        
        Args:
            train_scores: Metrics on training set
            test_scores: Metrics on test/holdout set
            cv_scores: Cross-validation scores
            dataset_info: Dataset characteristics
            
        Returns:
            List of overfitting warnings
        """
        self.warnings = []
        
        # Check 1: Train vs Test Gap
        self._check_train_test_gap(train_scores, test_scores)
        
        # Check 2: Unrealistic Perfect Scores
        self._check_perfect_scores(test_scores, dataset_info)
        
        # Check 3: CV Score Variance
        self._check_cv_variance(cv_scores)
        
        # Check 4: Small Test Set
        self._check_test_set_size(dataset_info)
        
        # Check 5: Imbalanced Data with Perfect Scores
        self._check_imbalanced_perfect(test_scores, dataset_info)
        
        return self.warnings
    
    def _check_train_test_gap(
        self, 
        train_scores: Dict[str, float], 
        test_scores: Dict[str, float]
    ):
        """Check if training accuracy significantly exceeds test accuracy."""
        for metric in ['accuracy', 'f1_macro']:
            if metric in train_scores and metric in test_scores:
                gap = train_scores[metric] - test_scores[metric]
                
                if gap > 0.15:  # 15% gap
                    self.warnings.append(OverfittingWarning(
                        severity='HIGH',
                        warning_type='TRAIN_TEST_GAP',
                        message=f"⚠️ **Severe Overfitting**: Training {metric} ({train_scores[metric]:.2%}) is {gap:.1%} higher than test {metric} ({test_scores[metric]:.2%})",
                        recommendations=[
                            "Your model memorized training data instead of learning patterns",
                            "Collect more diverse data (aim for 10x current size)",
                            "Use regularization: max_depth=5 for trees, C=0.1 for SVM",
                            "Try simpler models like LogisticRegression",
                            "Apply feature selection to top 10-20 features"
                        ]
                    ))
                elif gap > 0.10:  # 10% gap
                    self.warnings.append(OverfittingWarning(
                        severity='MEDIUM',
                        warning_type='TRAIN_TEST_GAP',
                        message=f"⚠️ **Moderate Overfitting**: Training {metric} is {gap:.1%} higher than test",
                        recommendations=[
                            "Consider using cross-validation",
                            "Try regularization to prevent overfitting",
                            "Reduce model complexity"
                        ]
                    ))
    
    def _check_perfect_scores(
        self, 
        test_scores: Dict[str, float],
        dataset_info: Dict[str, Any]
    ):
        """Check for unrealistic perfect or near-perfect scores."""
        accuracy = test_scores.get('accuracy', 0)
        n_samples = dataset_info.get('n_samples', 1000)
        
        if accuracy > 0.99 and n_samples < 500:
            self.warnings.append(OverfittingWarning(
                severity='HIGH',
                warning_type='PERFECT_SCORE_SMALL_DATA',
                message=f"🚨 **Data Leakage Suspected**: {accuracy:.1%} accuracy on {n_samples} samples is unrealistic",
                recommendations=[
                    "Verify train/test split is correct",
                    "Check for data leakage (features revealing target)",
                    "Try different random seed to test stability",
                    "Collect more data for reliable results",
                    "Simplify problem or verify data isn't too clean"
                ]
            ))
        elif accuracy > 0.95:
            self.warnings.append(OverfittingWarning(
                severity='MEDIUM',
                warning_type='VERY_HIGH_SCORE',
                message=f"⚠️ **Unusually High Accuracy**: {accuracy:.1%} - verify it's genuine",
                recommendations=[
                    "Test on new data from different time period",
                    "Check if problem is unusually easy",
                    "Verify no target leakage in features"
                ]
            ))
    
    def _check_cv_variance(self, cv_scores: Dict[str, List[float]]):
        """Check if CV scores have suspiciously low variance."""
        for metric, scores in cv_scores.items():
            if len(scores) >= 3:
                std = np.std(scores)
                mean = np.mean(scores)
                
                if std < 0.01 and mean > 0.9:
                    self.warnings.append(OverfittingWarning(
                        severity='MEDIUM',
                        warning_type='LOW_CV_VARIANCE',
                        message=f"🤔 **Suspiciously Consistent**: {metric} has {std:.3f} std across folds",
                        recommendations=[
                            "Low variance suggests problem might be too easy",
                            "Check for data duplication",
                            "Verify stratification is working"
                        ]
                    ))
    
    def _check_test_set_size(self, dataset_info: Dict[str, Any]):
        """Check if test set is too small."""
        n_test = dataset_info.get('n_test_samples', 0)
        n_classes = dataset_info.get('n_classes', 2)
        
        min_samples_per_class = n_test / n_classes if n_classes > 0 else n_test
        
        if n_test < 30:
            self.warnings.append(OverfittingWarning(
                severity='HIGH',
                warning_type='SMALL_TEST_SET',
                message=f"⚠️ **Test Set Too Small**: Only {n_test} samples is insufficient",
                recommendations=[
                    "Collect more data - aim for 100+ test samples",
                    "Use cross-validation instead of single split",
                    "Results are unreliable with this sample size"
                ]
            ))
        elif min_samples_per_class < 10:
            self.warnings.append(OverfittingWarning(
                severity='MEDIUM',
                warning_type='FEW_SAMPLES_PER_CLASS',
                message=f"⚠️ **Imbalanced Test Set**: ~{min_samples_per_class:.0f} samples per class",
                recommendations=[
                    "Use stratified sampling",
                    "Consider oversampling minority classes",
                    "Aim for 30+ samples per class"
                ]
            ))
    
    def _check_imbalanced_perfect(
        self,
        test_scores: Dict[str, float],
        dataset_info: Dict[str, Any]
    ):
        """Check for perfect scores on imbalanced data."""
        accuracy = test_scores.get('accuracy', 0)
        class_balance = dataset_info.get('class_balance', {})
        
        if class_balance:
            counts = list(class_balance.values())
            if len(counts) >= 2:
                imbalance_ratio = max(counts) / min(counts)
                
                if accuracy > 0.95 and imbalance_ratio > 3:
                    self.warnings.append(OverfittingWarning(
                        severity='HIGH',
                        warning_type='IMBALANCED_PERFECT',
                        message=f"🚨 **Misleading Accuracy**: {accuracy:.1%} on {imbalance_ratio:.1f}:1 imbalanced data",
                        recommendations=[
                            f"Accuracy is misleading - dummy model would get {max(counts):.1%}",
                            "Use F1-score, Precision, Recall instead",
                            "Check confusion matrix for majority class bias",
                            "Apply SMOTE or class weighting",
                            "Use balanced_accuracy_score"
                        ]
                    ))
    
    def get_user_guidance(self) -> Dict[str, Any]:
        """Generate user-friendly guidance document."""
        if not self.warnings:
            return {
                'has_issues': False,
                'message': "✅ No overfitting issues detected",
                'severity': 'NONE'
            }
        
        severities = [w.severity for w in self.warnings]
        overall_severity = 'HIGH' if 'HIGH' in severities else 'MEDIUM' if 'MEDIUM' in severities else 'LOW'
        
        return {
            'has_issues': True,
            'overall_severity': overall_severity,
            'warning_count': len(self.warnings),
            'warnings': [
                {
                    'severity': w.severity,
                    'type': w.warning_type,
                    'message': w.message,
                    'recommendations': w.recommendations
                }
                for w in self.warnings
            ],
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate executive summary."""
        high_count = sum(1 for w in self.warnings if w.severity == 'HIGH')
        medium_count = sum(1 for w in self.warnings if w.severity == 'MEDIUM')
        
        if high_count > 0:
            return f"🚨 **Critical Issues**: {high_count} high-severity warnings. Performance likely unrealistic."
        elif medium_count > 0:
            return f"⚠️ **Moderate Concerns**: {medium_count} potential issues. Results may not generalize."
        else:
            return "✅ Minor concerns. Review recommendations for best practices."
