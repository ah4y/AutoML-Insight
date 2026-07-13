"""
Overfitting Detection and User Guidance System.
Detects unrealistic model performance and provides actionable recommendations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


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
        dataset_info: Dict[str, Any],
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

    def _check_train_test_gap(self, train_scores: Dict[str, float], test_scores: Dict[str, float]):
        """Check if training accuracy significantly exceeds test accuracy."""
        for metric in ["accuracy", "f1_macro"]:
            if metric in train_scores and metric in test_scores:
                gap = train_scores[metric] - test_scores[metric]

                # FIXED: More realistic thresholds
                if gap > 0.20:  # 20% gap - severe
                    self.warnings.append(
                        OverfittingWarning(
                            severity="HIGH",
                            warning_type="SEVERE_OVERFITTING",
                            message=f"🚨 **Severe Overfitting**: Training {metric} ({train_scores[metric]:.2%}) is {gap:.1%} higher than test {metric} ({test_scores[metric]:.2%})",
                            recommendations=[
                                "Model has memorized training data instead of learning patterns",
                                "Collect more diverse data (aim for 5-10x current size)",
                                "Use strong regularization: max_depth=3-5 for trees, C=0.01-0.1 for SVM",
                                "Try simpler models like LogisticRegression or Naive Bayes",
                                "Apply feature selection to reduce dimensionality",
                            ],
                        )
                    )
                elif gap > 0.15:  # 15% gap - moderate to high
                    self.warnings.append(
                        OverfittingWarning(
                            severity="MEDIUM",
                            warning_type="MODERATE_OVERFITTING",
                            message=f"⚠️ **Moderate Overfitting**: Training {metric} is {gap:.1%} higher than test {metric}",
                            recommendations=[
                                "Model is overfitting but may still be usable",
                                "Consider cross-validation for more robust evaluation",
                                "Try regularization parameters",
                                "Collect more training data if possible",
                            ],
                        )
                    )
                elif gap > 0.10:  # 10-15% gap - informational
                    self.warnings.append(
                        OverfittingWarning(
                            severity="LOW",
                            warning_type="MINOR_OVERFITTING",
                            message=f"ℹ️ **Minor Overfitting**: Training {metric} is {gap:.1%} higher than test {metric}",
                            recommendations=[
                                "Some overfitting detected, but within acceptable range",
                                "Monitor performance on new data",
                                "Consider regularization for improvement",
                            ],
                        )
                    )

    def _check_perfect_scores(self, test_scores: Dict[str, float], dataset_info: Dict[str, Any]):
        """Check for unrealistic perfect or near-perfect scores."""
        accuracy = test_scores.get("accuracy", 0)
        f1_macro = test_scores.get("f1_macro", 0)
        n_samples = dataset_info.get("n_samples", 1000)
        n_test = dataset_info.get("n_test_samples", n_samples // 4)

        # FIXED: Only warn for 100% accuracy (truly perfect), not 95%+
        if accuracy >= 0.999 and n_test < 500:
            self.warnings.append(
                OverfittingWarning(
                    severity="HIGH",
                    warning_type="PERFECT_SCORE_SMALL_DATA",
                    message=f"🚨 **Data Leakage Suspected**: {accuracy:.1%} accuracy on {n_test} test samples is unrealistic",
                    recommendations=[
                        "Verify train/test split is correct",
                        "Check for data leakage (features revealing target)",
                        "Try different random seed to test stability",
                        "Collect more data for reliable results",
                        "Review feature engineering for leakage",
                    ],
                )
            )
        elif accuracy >= 0.999 and n_test >= 500:
            # Perfect score on large test set - likely data leakage
            self.warnings.append(
                OverfittingWarning(
                    severity="HIGH",
                    warning_type="PERFECT_SCORE_LEAKAGE",
                    message=f"🚨 **Data Leakage Likely**: {accuracy:.1%} accuracy on {n_test} samples suggests feature leakage",
                    recommendations=[
                        "Check for target leakage in features",
                        "Verify features don't contain future information",
                        "Review feature engineering process",
                        "Test on completely new data",
                    ],
                )
            )
        elif accuracy > 0.98 and f1_macro > 0.98 and n_test < 100:
            # Near-perfect on tiny test set
            self.warnings.append(
                OverfittingWarning(
                    severity="MEDIUM",
                    warning_type="VERY_HIGH_SCORE_SMALL",
                    message=f"⚠️ **High Score, Small Test Set**: {accuracy:.1%} accuracy on only {n_test} samples",
                    recommendations=[
                        "Test set is very small - results may not be reliable",
                        "Use cross-validation for more robust evaluation",
                        "Collect more test data if possible",
                    ],
                )
            )

    def _check_cv_variance(self, cv_scores: Dict[str, List[float]]):
        """Check if CV scores have suspiciously low variance."""
        for metric, scores in cv_scores.items():
            if len(scores) >= 3:
                std = np.std(scores)
                mean = np.mean(scores)

                if std < 0.01 and mean > 0.9:
                    self.warnings.append(
                        OverfittingWarning(
                            severity="MEDIUM",
                            warning_type="LOW_CV_VARIANCE",
                            message=f"🤔 **Suspiciously Consistent**: {metric} has {std:.3f} std across folds",
                            recommendations=[
                                "Low variance suggests problem might be too easy",
                                "Check for data duplication",
                                "Verify stratification is working",
                            ],
                        )
                    )

    def _check_test_set_size(self, dataset_info: Dict[str, Any]):
        """Check if test set is too small."""
        n_test = dataset_info.get("n_test_samples", 0)
        n_classes = dataset_info.get("n_classes", 2)

        min_samples_per_class = n_test / n_classes if n_classes > 0 else n_test

        if n_test < 30:
            self.warnings.append(
                OverfittingWarning(
                    severity="HIGH",
                    warning_type="SMALL_TEST_SET",
                    message=f"⚠️ **Test Set Too Small**: Only {n_test} samples is insufficient",
                    recommendations=[
                        "Collect more data - aim for 100+ test samples",
                        "Use cross-validation instead of single split",
                        "Results are unreliable with this sample size",
                    ],
                )
            )
        elif min_samples_per_class < 10:
            self.warnings.append(
                OverfittingWarning(
                    severity="MEDIUM",
                    warning_type="FEW_SAMPLES_PER_CLASS",
                    message=f"⚠️ **Imbalanced Test Set**: ~{min_samples_per_class:.0f} samples per class",
                    recommendations=[
                        "Use stratified sampling",
                        "Consider oversampling minority classes",
                        "Aim for 30+ samples per class",
                    ],
                )
            )

    def _check_imbalanced_perfect(self, test_scores: Dict[str, float], dataset_info: Dict[str, Any]):
        """Check for perfect scores on imbalanced data."""
        accuracy = test_scores.get("accuracy", 0)
        f1_macro = test_scores.get("f1_macro", 0)
        class_balance = dataset_info.get("class_balance", {})

        if class_balance:
            counts = list(class_balance.values())
            if len(counts) >= 2:
                imbalance_ratio = max(counts) / min(counts)
                majority_baseline = max(counts)

                # FIXED: Only warn if accuracy is high BUT F1 is low (indicating majority class bias)
                # If F1 is also high, the model is genuinely good, not just predicting majority
                if imbalance_ratio > 3:
                    # Check if model is just predicting majority class
                    if accuracy > 0.90 and f1_macro < 0.70:
                        # High accuracy but low F1 = majority class bias
                        self.warnings.append(
                            OverfittingWarning(
                                severity="HIGH",
                                warning_type="MAJORITY_CLASS_BIAS",
                                message=f"🚨 **Majority Class Bias**: {accuracy:.1%} accuracy but {f1_macro:.1%} F1-score on {imbalance_ratio:.1f}:1 imbalanced data",
                                recommendations=[
                                    f"Model is likely just predicting majority class (baseline: {majority_baseline:.1%})",
                                    "Use F1-score, Precision, Recall instead of accuracy",
                                    "Check confusion matrix to confirm",
                                    "Apply SMOTE or class weighting",
                                    "Use balanced_accuracy_score",
                                ],
                            )
                        )
                    elif accuracy > 0.95 and f1_macro > 0.90:
                        # High accuracy AND high F1 = genuinely good, just note the imbalance
                        self.warnings.append(
                            OverfittingWarning(
                                severity="LOW",
                                warning_type="IMBALANCED_DATA_NOTE",
                                message=f"ℹ️ **Note**: High performance on {imbalance_ratio:.1f}:1 imbalanced data - verify with confusion matrix",
                                recommendations=[
                                    "Model appears to handle imbalance well (high F1-score)",
                                    "Still verify confusion matrix shows good per-class performance",
                                    "Consider testing on new data to confirm",
                                ],
                            )
                        )
                    elif accuracy > 0.85:
                        # Moderate accuracy on imbalanced data - informational
                        self.warnings.append(
                            OverfittingWarning(
                                severity="LOW",
                                warning_type="IMBALANCED_DATA_INFO",
                                message=f"ℹ️ **Imbalanced Data**: {imbalance_ratio:.1f}:1 class ratio detected",
                                recommendations=[
                                    f"Baseline accuracy (always predict majority): {majority_baseline:.1%}",
                                    "Focus on F1-score, Precision, Recall metrics",
                                    "Check confusion matrix for per-class performance",
                                ],
                            )
                        )

    def get_user_guidance(self) -> Dict[str, Any]:
        """Generate user-friendly guidance document."""
        if not self.warnings:
            return {"has_issues": False, "message": "✅ No overfitting issues detected", "severity": "NONE"}

        severities = [w.severity for w in self.warnings]
        overall_severity = "HIGH" if "HIGH" in severities else "MEDIUM" if "MEDIUM" in severities else "LOW"

        return {
            "has_issues": True,
            "overall_severity": overall_severity,
            "warning_count": len(self.warnings),
            "warnings": [
                {
                    "severity": w.severity,
                    "type": w.warning_type,
                    "message": w.message,
                    "recommendations": w.recommendations,
                }
                for w in self.warnings
            ],
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        high_count = sum(1 for w in self.warnings if w.severity == "HIGH")
        medium_count = sum(1 for w in self.warnings if w.severity == "MEDIUM")

        if high_count > 0:
            return f"🚨 **Critical Issues**: {high_count} high-severity warnings. Performance likely unrealistic."
        elif medium_count > 0:
            return f"⚠️ **Moderate Concerns**: {medium_count} potential issues. Results may not generalize."
        else:
            return "✅ Minor concerns. Review recommendations for best practices."
