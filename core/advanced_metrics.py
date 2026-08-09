"""Advanced Classification Metrics Beyond Basic Accuracy/Precision/Recall.

This module provides extended metrics for comprehensive model evaluation including
calibration analysis, confidence metrics, and detailed confusion matrix breakdowns.
"""

import logging

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


class AdvancedMetricsCalculator:
    """Calculate extended classification metrics and calibration analysis.

    This class provides methods to compute advanced metrics that go beyond basic
    accuracy/precision/recall, including model calibration, prediction confidence
    analysis, and detailed confusion matrix breakdowns.
    """

    def __init__(self):
        """Initialize the AdvancedMetricsCalculator."""
        logger.info("AdvancedMetricsCalculator initialized")

    def compute_extended_metrics(self, y_true, y_pred, y_proba=None):
        """Compute extended classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for probability-based metrics)

        Returns:
            Dictionary with extended metrics:
                - matthews_corrcoef: MCC score (-1 to 1)
                - cohen_kappa: Cohen's Kappa score
                - balanced_accuracy: Balanced accuracy score
                - jaccard_score: Jaccard similarity coefficient
                - hamming_loss: Hamming loss
        """
        metrics = {}

        try:
            # Matthews Correlation Coefficient
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                metrics["matthews_corrcoef"] = float(mcc)
                logger.info(f"Matthews Correlation Coefficient: {mcc:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute MCC: {e}")
                metrics["matthews_corrcoef"] = None

            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(y_true, y_pred)
                metrics["cohen_kappa"] = float(kappa)
                logger.info(f"Cohen's Kappa: {kappa:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute Cohen's Kappa: {e}")
                metrics["cohen_kappa"] = None

            # Balanced Accuracy
            try:
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                metrics["balanced_accuracy"] = float(balanced_acc)
                logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute Balanced Accuracy: {e}")
                metrics["balanced_accuracy"] = None

            # Jaccard Score
            try:
                # Use average='weighted' for multi-class
                n_classes = len(np.unique(y_true))
                avg_method = "binary" if n_classes == 2 else "weighted"
                jaccard = jaccard_score(y_true, y_pred, average=avg_method)
                metrics["jaccard_score"] = float(jaccard)
                logger.info(f"Jaccard Score: {jaccard:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute Jaccard Score: {e}")
                metrics["jaccard_score"] = None

            # Hamming Loss
            try:
                hamming = hamming_loss(y_true, y_pred)
                metrics["hamming_loss"] = float(hamming)
                logger.info(f"Hamming Loss: {hamming:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute Hamming Loss: {e}")
                metrics["hamming_loss"] = None

            logger.info(f"Computed {len([v for v in metrics.values() if v is not None])} extended metrics")

        except Exception as e:
            logger.error(f"Error computing extended metrics: {e}")

        return metrics

    def compute_calibration_metrics(self, y_true, y_proba, n_bins=10):
        """Compute calibration metrics for probability predictions.

        Calibration measures how well predicted probabilities match actual frequencies.
        A well-calibrated model's predicted probabilities should reflect true likelihoods.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities (n_samples, n_classes)
            n_bins: Number of bins for calibration curve (default: 10)

        Returns:
            Dictionary with calibration data per class:
                - ece: Expected Calibration Error
                - per_class: List of dicts with prob_true, prob_pred for each class
        """
        try:
            logger.info("Computing calibration metrics")

            if y_proba is None or len(y_proba.shape) != 2:
                logger.warning("Invalid probability array for calibration")
                return None

            n_classes = y_proba.shape[1]
            calibration_data = {"ece": None, "per_class": []}

            # Compute ECE (Expected Calibration Error)
            ece_sum = 0.0
            total_samples = 0

            for class_idx in range(n_classes):
                # Get probabilities and binary labels for this class
                probs = y_proba[:, class_idx]
                y_binary = (y_true == class_idx).astype(int)

                # Bin the probabilities
                bins = np.linspace(0, 1, n_bins + 1)
                bin_indices = np.digitize(probs, bins[:-1]) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                prob_true = []
                prob_pred = []

                for bin_idx in range(n_bins):
                    mask = bin_indices == bin_idx
                    if mask.sum() > 0:
                        bin_accuracy = y_binary[mask].mean()
                        bin_confidence = probs[mask].mean()
                        prob_true.append(float(bin_accuracy))
                        prob_pred.append(float(bin_confidence))

                        # Accumulate ECE
                        ece_sum += mask.sum() * abs(bin_accuracy - bin_confidence)
                        total_samples += mask.sum()
                    else:
                        prob_true.append(None)
                        prob_pred.append(None)

                calibration_data["per_class"].append(
                    {"class_idx": int(class_idx), "prob_true": prob_true, "prob_pred": prob_pred}
                )

            # Calculate overall ECE
            if total_samples > 0:
                calibration_data["ece"] = float(ece_sum / total_samples)
                logger.info(f"Expected Calibration Error (ECE): {calibration_data['ece']:.4f}")

            return calibration_data

        except Exception as e:
            logger.error(f"Error computing calibration metrics: {e}")
            return None

    def compute_prediction_confidence_analysis(self, y_true, y_pred, y_proba):
        """Analyze prediction confidence and its relationship with correctness.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (n_samples, n_classes)

        Returns:
            Dictionary with confidence analysis:
                - avg_confidence_correct: Average confidence for correct predictions
                - avg_confidence_incorrect: Average confidence for incorrect predictions
                - confidence_separation: Difference between correct/incorrect confidence
                - low_confidence_errors: Count of errors with confidence < 0.6
                - high_confidence_errors: Count of errors with confidence > 0.8
        """
        try:
            logger.info("Computing prediction confidence analysis")

            if y_proba is None or len(y_proba.shape) != 2:
                logger.warning("Invalid probability array for confidence analysis")
                return None

            # Get confidence (max probability) for each prediction
            confidences = np.max(y_proba, axis=1)

            # Identify correct and incorrect predictions
            correct_mask = y_true == y_pred

            # Compute average confidences
            avg_confidence_correct = float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0
            avg_confidence_incorrect = float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0

            # Confidence separation (larger is better)
            confidence_separation = avg_confidence_correct - avg_confidence_incorrect

            # Count errors by confidence level
            incorrect_confidences = confidences[~correct_mask]
            low_confidence_errors = int((incorrect_confidences < 0.6).sum()) if len(incorrect_confidences) > 0 else 0
            high_confidence_errors = int((incorrect_confidences > 0.8).sum()) if len(incorrect_confidences) > 0 else 0

            logger.info(f"Avg confidence (correct): {avg_confidence_correct:.4f}")
            logger.info(f"Avg confidence (incorrect): {avg_confidence_incorrect:.4f}")
            logger.info(f"Confidence separation: {confidence_separation:.4f}")
            logger.info(f"High-confidence errors: {high_confidence_errors}")

            return {
                "avg_confidence_correct": avg_confidence_correct,
                "avg_confidence_incorrect": avg_confidence_incorrect,
                "confidence_separation": confidence_separation,
                "low_confidence_errors": low_confidence_errors,
                "high_confidence_errors": high_confidence_errors,
            }

        except Exception as e:
            logger.error(f"Error computing confidence analysis: {e}")
            return None

    def analyze_confusion_matrix_detailed(self, y_true, y_pred, class_names=None):
        """Perform detailed confusion matrix analysis with per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names (optional)

        Returns:
            Dictionary with detailed analysis:
                - per_class_precision: Dict of precision per class
                - per_class_recall: Dict of recall per class
                - per_class_f1: Dict of F1 score per class
                - misclassification_patterns: Top misclassification pairs
        """
        try:
            logger.info("Performing detailed confusion matrix analysis")

            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(classes)

            if class_names is None:
                class_names = [f"Class {i}" for i in classes]

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=classes)

            # Per-class metrics
            per_class_precision = {}
            per_class_recall = {}
            per_class_f1 = {}

            for idx, class_label in enumerate(classes):
                class_name = class_names[idx] if idx < len(class_names) else f"Class {class_label}"

                # True Positives, False Positives, False Negatives
                tp = cm[idx, idx]
                fp = cm[:, idx].sum() - tp
                fn = cm[idx, :].sum() - tp

                # Precision
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                per_class_precision[class_name] = float(precision)

                # Recall
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                per_class_recall[class_name] = float(recall)

                # F1 Score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                per_class_f1[class_name] = float(f1)

            # Find top misclassification patterns
            misclassification_patterns = []
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j and cm[i, j] > 0:
                        true_class = class_names[i] if i < len(class_names) else f"Class {classes[i]}"
                        pred_class = class_names[j] if j < len(class_names) else f"Class {classes[j]}"
                        misclassification_patterns.append(
                            {"true_class": true_class, "predicted_class": pred_class, "count": int(cm[i, j])}
                        )

            # Sort by count (descending)
            misclassification_patterns.sort(key=lambda x: x["count"], reverse=True)

            logger.info(f"Analyzed {n_classes} classes")
            logger.info(f"Found {len(misclassification_patterns)} misclassification patterns")

            return {
                "per_class_precision": per_class_precision,
                "per_class_recall": per_class_recall,
                "per_class_f1": per_class_f1,
                "misclassification_patterns": misclassification_patterns[:10],  # Top 10
            }

        except Exception as e:
            logger.error(f"Error in detailed confusion matrix analysis: {e}")
            return None
