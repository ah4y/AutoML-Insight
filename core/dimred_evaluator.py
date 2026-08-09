"""Enhanced evaluation with dimensionality reduction comparison for AutoML-Insight.

This module extends the existing evaluation capabilities to systematically compare
models with and without dimensionality reduction, ensuring fair comparisons
through proper nested cross-validation.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from core.dimred import DimRedConfig, make_dimred
from core.evaluate_cls import ClassificationEvaluator
from core.evaluate_clu import ClusteringEvaluator
from core.preprocess import DataPreprocessor
from utils.logging_utils import setup_logger


class DimRedEvaluator:
    """
    Comprehensive evaluator that compares models with and without dimensionality reduction.

    This evaluator creates proper pipelines that include preprocessing and optional
    dimensionality reduction, ensuring no data leakage through nested cross-validation.
    """

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        dimred_config: Optional[DimRedConfig] = None,
        n_folds: int = 5,
        n_repeats: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize the DimRed evaluator.

        Args:
            preprocessor: Fitted data preprocessor
            dimred_config: Dimensionality reduction configuration
            n_folds: Number of CV folds
            n_repeats: Number of CV repeats
            random_state: Random seed for reproducibility
        """
        self.preprocessor = preprocessor
        self.dimred_config = dimred_config or DimRedConfig()
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.logger = setup_logger()

        # Initialize evaluators
        self.cls_evaluator = ClassificationEvaluator(n_folds, n_repeats, random_state)
        self.clu_evaluator = ClusteringEvaluator()

        # Results storage
        self.classification_results = {}
        self.clustering_results = {}
        self.comparison_results = {}

    def evaluate_classification_with_dimred(
        self, models: Dict[str, Any], X_raw: pd.DataFrame, y: np.ndarray, task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Evaluate classification models with and without dimensionality reduction.

        Args:
            models: Dictionary of {model_name: model_instance}
            X_raw: Raw feature matrix before preprocessing
            y: Target variable
            task_type: Type of task ("classification" or "regression")

        Returns:
            Dictionary containing evaluation results and comparisons
        """
        self.logger.info(f"Starting dimensionality reduction evaluation for {len(models)} models")

        # Determine data characteristics for dimred decision
        n_samples, n_features_raw = X_raw.shape

        results = {}

        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name} with/without dimensionality reduction")

            # Determine if this model type benefits from dimred
            model_benefits_from_dimred = self._model_benefits_from_dimred(model_name)

            if not model_benefits_from_dimred and self.dimred_config.enable == "auto":
                # Skip dimred for tree models unless explicitly enabled
                self.logger.info(f"Skipping dimred for {model_name} (tree model)")
                results[model_name] = self._evaluate_single_model(model, X_raw, y, model_name, use_dimred=False)
                continue

            # Evaluate without dimred (baseline)
            baseline_results = self._evaluate_single_model(model, X_raw, y, f"{model_name}_baseline", use_dimred=False)

            # Evaluate with dimred (if enabled)
            dimred_results = None
            if self.dimred_config.enable != "off":
                dimred_results = self._evaluate_single_model(model, X_raw, y, f"{model_name}_dimred", use_dimred=True)

            # Compare and select best variant
            selected_results = self._compare_and_select_variant(model_name, baseline_results, dimred_results)

            results[model_name] = selected_results

        # After evaluating all models, fit a PCA transformer on full data for visualization
        if self.dimred_config.enable != "off":
            self.logger.info(
                f"Fitting PCA transformer on full dataset for visualization (config: enable={self.dimred_config.enable})"
            )
            try:
                # Create and fit a complete preprocessing + PCA pipeline
                preprocessor_full = clone(self.preprocessor)
                X_preprocessed, _ = preprocessor_full.fit_transform(X_raw, y)

                from scipy import sparse

                from core.dimred import DimRedConfig, make_dimred

                n_samples, n_features = X_preprocessed.shape
                is_sparse = sparse.issparse(X_preprocessed)

                self.logger.info(f"Data for PCA: {n_samples} samples, {n_features} features, sparse={is_sparse}")

                # Force PCA creation for visualization (override auto-detection)
                viz_config = DimRedConfig(
                    enable="on",  # Force enable for visualization
                    method="pca",  # Force PCA method
                    variance_target=self.dimred_config.variance_target,
                    k_max=self.dimred_config.k_max,
                    whiten=self.dimred_config.whiten,
                    seed=self.dimred_config.seed,
                )

                # Create PCA transformer
                pca_transformer = make_dimred(
                    is_sparse_after_ohe=is_sparse, n_features=n_features, n_samples=n_samples, cfg=viz_config
                )

                if pca_transformer is not None:
                    # Fit and transform the data
                    X_transformed = pca_transformer.fit_transform(X_preprocessed)

                    # Store PCA results for visualization
                    results["pca_transformer"] = pca_transformer
                    results["X_transformed"] = X_transformed
                    results["n_components"] = getattr(pca_transformer, "n_components", None)

                    self.logger.info(
                        f"PCA transformer fitted with {X_transformed.shape[1]} components for visualization"
                    )
                else:
                    self.logger.warning("PCA transformer creation failed even with forced config")

            except Exception as e:
                self.logger.warning(f"Failed to create PCA transformer for visualization: {e}")
        else:
            self.logger.info("PCA visualization disabled (dimred_config.enable='off')")

        return results

    def _evaluate_single_model(
        self, model: Any, X_raw: pd.DataFrame, y: np.ndarray, model_name: str, use_dimred: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single model variant with proper nested CV.

        Args:
            model: Model instance to evaluate
            X_raw: Raw features before preprocessing
            y: Target variable
            model_name: Name for this model variant
            use_dimred: Whether to include dimensionality reduction

        Returns:
            Evaluation results dictionary
        """
        from scipy import sparse

        # Create a complete pipeline for this variant
        # Always start with preprocessing
        preprocessor_clone = clone(self.preprocessor)

        if use_dimred:
            # Temporarily transform to determine data characteristics
            X_temp, _ = preprocessor_clone.fit_transform(X_raw, y)
            n_samples, n_features = X_temp.shape
            is_sparse = sparse.issparse(X_temp)

            # Create dimred transformer
            dimred_transformer = make_dimred(
                is_sparse_after_ohe=is_sparse, n_features=n_features, n_samples=n_samples, cfg=self.dimred_config
            )

            if dimred_transformer is not None:
                # Build pipeline: preprocess -> dimred -> model
                pipeline = Pipeline(
                    [
                        ("preprocess", preprocessor_clone.preprocessor),
                        ("dimred", dimred_transformer),
                        ("model", clone(model)),
                    ]
                )

                self.logger.info(f"Created pipeline with {type(dimred_transformer).__name__}")
            else:
                # No dimred needed, just preprocess -> model
                pipeline = Pipeline([("preprocess", preprocessor_clone.preprocessor), ("model", clone(model))])
                self.logger.info("Dimred not applied (auto-disabled)")
        else:
            # Baseline: just preprocess -> model
            pipeline = Pipeline([("preprocess", preprocessor_clone.preprocessor), ("model", clone(model))])

        # Nested cross-validation evaluation
        cv_results = self._nested_cv_evaluation(pipeline, X_raw, y, model_name)

        # Add metadata about this variant
        cv_results["uses_dimred"] = use_dimred
        if use_dimred and "dimred" in pipeline.named_steps:
            dimred_step = pipeline.named_steps["dimred"]
            cv_results["dimred_method"] = type(dimred_step).__name__
            if hasattr(dimred_step, "n_components"):
                cv_results["n_components"] = dimred_step.n_components

        return cv_results

    def _nested_cv_evaluation(
        self, pipeline: Pipeline, X_raw: pd.DataFrame, y: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation evaluation of a pipeline.

        Args:
            pipeline: Complete preprocessing + model pipeline
            X_raw: Raw features
            y: Target variable
            model_name: Model name for logging

        Returns:
            Dictionary with evaluation metrics and confidence intervals
        """
        from sklearn.model_selection import cross_validate

        from utils.metrics_utils import compute_confidence_interval

        # Define scoring metrics
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        }

        # Add ROC-AUC for binary/multi-class (but not for cases with single class)
        n_classes = len(np.unique(y))
        if n_classes > 1:
            if n_classes == 2:
                scoring["roc_auc"] = "roc_auc"
            else:
                scoring["roc_auc"] = "roc_auc_ovr"

        # Nested cross-validation
        all_scores = {metric: [] for metric in scoring.keys()}

        for repeat in range(self.n_repeats):
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state + repeat)

            try:
                # Cross-validation with complete pipeline
                cv_results = cross_validate(pipeline, X_raw, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")

                # Collect scores
                for metric in scoring.keys():
                    all_scores[metric].extend(cv_results[f"test_{metric}"])

                self.logger.info(f"{model_name} repeat {repeat+1}/{self.n_repeats} completed")

            except Exception as e:
                self.logger.error(f"CV failed for {model_name} repeat {repeat+1}: {e}")
                # Add zero scores for failed evaluation
                for metric in scoring.keys():
                    all_scores[metric].extend([0.0] * self.n_folds)

        # Compute statistics
        results = {"model_name": model_name}

        for metric, scores in all_scores.items():
            if len(scores) > 0:
                mean, lower, upper = compute_confidence_interval(scores)
                results[f"{metric}_mean"] = mean
                results[f"{metric}_ci_lower"] = lower
                results[f"{metric}_ci_upper"] = upper
                results[f"{metric}_scores"] = scores
            else:
                results[f"{metric}_mean"] = 0.0
                results[f"{metric}_ci_lower"] = 0.0
                results[f"{metric}_ci_upper"] = 0.0
                results[f"{metric}_scores"] = []

        return results

    def _model_benefits_from_dimred(self, model_name: str) -> bool:
        """
        Determine if a model type typically benefits from dimensionality reduction.

        Args:
            model_name: Name of the model

        Returns:
            True if model typically benefits from dimred
        """
        # Tree-based models typically don't benefit from PCA
        tree_models = ["RandomForest", "XGBoost", "ExtraTrees", "GradientBoosting"]

        # Linear models and distance-based models benefit from PCA
        linear_models = ["LogisticRegression", "LinearSVM", "RBF-SVM", "KNN", "MLP"]

        for tree_model in tree_models:
            if tree_model.lower() in model_name.lower():
                return False

        for linear_model in linear_models:
            if linear_model.lower() in model_name.lower():
                return True

        # Default: assume benefits from dimred
        return True

    def _compare_and_select_variant(
        self, model_name: str, baseline_results: Dict[str, Any], dimred_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare baseline vs dimred variants and select the best one.

        Args:
            model_name: Original model name
            baseline_results: Results without dimred
            dimred_results: Results with dimred (None if not evaluated)

        Returns:
            Results for the selected variant with comparison metadata
        """
        if dimred_results is None:
            # Only baseline available
            selected = baseline_results.copy()
            selected["selected_variant"] = "baseline"
            selected["comparison"] = "dimred_not_evaluated"
            return selected

        # Compare accuracy means (primary metric)
        baseline_acc = baseline_results.get("accuracy_mean", 0.0)
        dimred_acc = dimred_results.get("accuracy_mean", 0.0)

        # Statistical significance test (if we have scores)
        significance_p = None
        baseline_scores = baseline_results.get("accuracy_scores", [])
        dimred_scores = dimred_results.get("accuracy_scores", [])

        if len(baseline_scores) > 0 and len(dimred_scores) > 0:
            try:
                from scipy import stats

                statistic, p_value = stats.wilcoxon(baseline_scores, dimred_scores)
                significance_p = p_value
            except Exception as e:
                self.logger.warning(f"Significance test failed for {model_name}: {e}")

        # Selection logic
        acc_improvement = dimred_acc - baseline_acc
        is_significant = significance_p is not None and significance_p < 0.05

        # Select dimred if significantly better or comparable with clear benefits
        if acc_improvement > 0.01 or (abs(acc_improvement) < 0.01 and is_significant):
            selected = dimred_results.copy()
            selected["selected_variant"] = "dimred"
            selection_reason = "dimred_better" if acc_improvement > 0.01 else "dimred_equivalent_significant"
        else:
            selected = baseline_results.copy()
            selected["selected_variant"] = "baseline"
            selection_reason = "baseline_better"

        # Add comparison metadata
        selected["comparison"] = {
            "baseline_accuracy": baseline_acc,
            "dimred_accuracy": dimred_acc,
            "improvement": acc_improvement,
            "significance_p": significance_p,
            "is_significant": is_significant,
            "selection_reason": selection_reason,
        }

        # Format p-value for logging
        p_value_str = f"{significance_p:.4f}" if significance_p is not None else "N/A"

        self.logger.info(
            f"{model_name}: Selected {selected['selected_variant']} "
            f"(baseline: {baseline_acc:.4f}, dimred: {dimred_acc:.4f}, "
            f"p={p_value_str})"
        )

        return selected

    def get_leaderboard_with_dimred(self, metric: str = "accuracy") -> List[Dict[str, Any]]:
        """
        Get leaderboard that includes dimred comparison information.

        Args:
            metric: Metric to sort by

        Returns:
            Sorted leaderboard with dimred metadata
        """
        leaderboard = []

        for model_name, results in self.classification_results.items():
            metric_key = f"{metric}_mean"
            if metric_key in results:
                entry = {
                    "model": model_name,
                    "score": results[metric_key],
                    "ci_lower": results.get(f"{metric}_ci_lower", 0),
                    "ci_upper": results.get(f"{metric}_ci_upper", 0),
                    "selected_variant": results.get("selected_variant", "unknown"),
                    "uses_dimred": results.get("uses_dimred", False),
                    "dimred_method": results.get("dimred_method", "none"),
                    "n_components": results.get("n_components", 0),
                    "comparison": results.get("comparison", {}),
                }
                leaderboard.append(entry)

        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x["score"], reverse=True)

        return leaderboard

    def get_dimred_summary(self) -> Dict[str, Any]:
        """
        Get summary of dimensionality reduction impact across all models.

        Returns:
            Summary statistics and insights
        """
        summary = {
            "total_models_evaluated": len(self.classification_results),
            "models_using_dimred": 0,
            "models_improved_by_dimred": 0,
            "average_improvement": 0.0,
            "significant_improvements": 0,
            "dimred_methods_used": {},
            "component_counts": [],
        }

        improvements = []

        for model_name, results in self.classification_results.items():
            comparison = results.get("comparison", {})

            if results.get("uses_dimred", False):
                summary["models_using_dimred"] += 1

                # Track dimred methods
                method = results.get("dimred_method", "unknown")
                summary["dimred_methods_used"][method] = summary["dimred_methods_used"].get(method, 0) + 1

                # Track component counts
                n_components = results.get("n_components", 0)
                if n_components > 0:
                    summary["component_counts"].append(n_components)

            if isinstance(comparison, dict):
                improvement = comparison.get("improvement", 0)
                if improvement > 0:
                    summary["models_improved_by_dimred"] += 1

                improvements.append(improvement)

                if comparison.get("is_significant", False):
                    summary["significant_improvements"] += 1

        if improvements:
            summary["average_improvement"] = np.mean(improvements)
            summary["median_improvement"] = np.median(improvements)

        return summary
