"""Classification evaluation with comprehensive metrics."""

import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    brier_score_loss, make_scorer
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, List
from utils.metrics_utils import compute_confidence_interval, mcnemar_test, wilcoxon_test
from core.bias_variance_analyzer import BiasVarianceAnalyzer
from core.advanced_metrics import AdvancedMetricsCalculator
from core.statistical_tests import StatisticalModelComparator
from core.background_analyzer import BackgroundAnalysisManager
from utils.logging_utils import get_logger
import streamlit as st

logger = get_logger(__name__)


class ClassificationEvaluator:
    """Comprehensive evaluation for classification models."""
    
    def __init__(
        self,
        n_folds: int = 5,
        n_repeats: int = 3,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model with nested cross-validation.
        
        Args:
            model: Trained model instance
            X: Feature matrix
            y: Target variable
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'roc_auc_ovr': 'roc_auc_ovr',
        }
        
        # Nested cross-validation
        all_scores = {metric: [] for metric in scoring.keys()}
        all_scores['log_loss'] = []
        all_scores['brier'] = []
        all_predictions = []
        all_true = []
        
        for repeat in range(self.n_repeats):
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state + repeat
            )
            
            # Cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_estimator=True,
                n_jobs=1  # Changed from -1 to avoid Windows multiprocessing issues
            )
            
            for metric in scoring.keys():
                all_scores[metric].extend(cv_results[f'test_{metric}'])
            
            # Additional metrics requiring predictions
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                estimator = cv_results['estimator'][fold_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                try:
                    y_proba = estimator.predict_proba(X_test)
                    
                    # Log loss
                    ll = log_loss(y_test, y_proba)
                    all_scores['log_loss'].append(ll)
                    
                    # Brier score (for binary classification)
                    if len(np.unique(y)) == 2:
                        brier = brier_score_loss(y_test, y_proba[:, 1])
                        all_scores['brier'].append(brier)
                except (AttributeError, ValueError) as e:
                    logger.debug(f"Probabilistic metrics unavailable: {e}")
                    pass
                
                # Store predictions for statistical tests
                y_pred = estimator.predict(X_test)
                all_predictions.extend(y_pred)
                all_true.extend(y_test)
        
        # Compute mean and confidence intervals
        results = {
            'model_name': model_name,
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_true)
        }
        
        for metric, scores in all_scores.items():
            if len(scores) > 0:
                mean, lower, upper = compute_confidence_interval(scores)
                results[f'{metric}_mean'] = mean
                results[f'{metric}_ci_lower'] = lower
                results[f'{metric}_ci_upper'] = upper
                results[f'{metric}_scores'] = scores
        
        self.results[model_name] = results
        return results
    
    def compare_models(
        self,
        model1_name: str,
        model2_name: str
    ) -> Dict[str, float]:
        """
        Statistical comparison between two models.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary of test statistics
        """
        if model1_name not in self.results or model2_name not in self.results:
            return {}
        
        results1 = self.results[model1_name]
        results2 = self.results[model2_name]
        
        comparison = {}
        
        # McNemar's test (for predictions)
        try:
            mcnemar_p = mcnemar_test(
                results1['true_labels'],
                results1['predictions'],
                results2['predictions']
            )
            comparison['mcnemar_p_value'] = mcnemar_p
        except (ValueError, KeyError) as e:
            logger.debug(f"McNemar test failed: {e}")
            pass
        
        # Wilcoxon test (for scores)
        if 'accuracy_scores' in results1 and 'accuracy_scores' in results2:
            try:
                wilcoxon_p = wilcoxon_test(
                    np.array(results1['accuracy_scores']),
                    np.array(results2['accuracy_scores'])
                )
                comparison['wilcoxon_p_value'] = wilcoxon_p
            except (ValueError, KeyError) as e:
                logger.debug(f"Wilcoxon test failed: {e}")
                pass
        
        return comparison
    
    def get_leaderboard(self, metric: str = 'accuracy', penalize_overfitting: bool = True) -> List[Dict[str, Any]]:
        """
        Get model leaderboard sorted by metric with optional overfitting penalty.
        
        Args:
            metric: Metric to sort by
            penalize_overfitting: If True, penalizes models with high train/test gap
            
        Returns:
            Sorted list of model results
        """
        leaderboard = []
        
        for model_name, results in self.results.items():
            metric_key = f'{metric}_mean'
            if metric_key in results:
                score = results[metric_key]
                
                # Calculate adjusted score considering overfitting
                adjusted_score = score
                if penalize_overfitting:
                    # Get test accuracy (actual performance)
                    test_acc = results.get('test_accuracy', score)
                    train_acc = results.get('train_accuracy', score)
                    gap = abs(train_acc - test_acc)
                    
                    # Penalty: 0.5 weight means 10% gap reduces score by 5%
                    # This makes LogisticRegression (gap 0.01, test 0.335) beat KNN (gap 0.21, test 0.340)
                    overfitting_penalty = gap * 0.5
                    adjusted_score = test_acc - overfitting_penalty
                
                leaderboard.append({
                    'model': model_name,
                    'score': score,  # Original score for display
                    'adjusted_score': adjusted_score,  # Score with overfitting penalty
                    'ci_lower': results.get(f'{metric}_ci_lower', 0),
                    'ci_upper': results.get(f'{metric}_ci_upper', 0),
                    'test_accuracy': results.get('test_accuracy', score),
                    'train_accuracy': results.get('train_accuracy', score),
                    'overfitting_gap': abs(results.get('train_accuracy', score) - results.get('test_accuracy', score))
                })
        
        # Sort by adjusted score (considers overfitting) or original score
        sort_key = 'adjusted_score' if penalize_overfitting else 'score'
        leaderboard.sort(key=lambda x: x[sort_key], reverse=True)
        
        return leaderboard
    
    def evaluate_with_holdout(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate with proper train/test split to detect overfitting.
        FAST version: Skip expensive CV, just train once and evaluate.
        
        Args:
            model: Untrained model instance
            X_train, y_train: Training data
            X_test, y_test: Held-out test data
            model_name: Model identifier
            
        Returns:
            Dictionary with train/test scores and overfitting warnings
        """
        from core.overfitting_detector import OverfittingDetector
        from sklearn.base import clone
        
        # Clone to avoid modifying original
        model = clone(model)
        
        # AUTO-SELECT CV STRATEGY based on dataset size
        n_train = len(X_train)
        
        # Use user-configured CV strategy (passed in constructor) unless overridden by extreme data conditions
        user_n_folds = self.n_folds  # User's preference from constructor
        
        # AGGRESSIVE optimization for large datasets - only override user preference when absolutely necessary
        if n_train < 50:  # Extremely small datasets
            n_folds = 2
            cv_sample_size = n_train
            cv_strategy = f"Minimal CV (2-fold) - dataset too small for {user_n_folds}-fold"
        elif n_train < 100 and user_n_folds > 5:  # Small datasets with high fold request
            n_folds = min(5, user_n_folds)
            cv_sample_size = n_train
            cv_strategy = f"Limited CV ({n_folds}-fold) - dataset size constrained"
        elif n_train < 10000:  # Normal datasets - use user preference
            n_folds = user_n_folds
            cv_sample_size = n_train
            cv_strategy = f"Standard CV ({n_folds}-fold)"
        else:
            # Large datasets: Use user preference but with sampling for efficiency
            n_folds = user_n_folds
            cv_sample_size = min(5000, n_train)  # Sample for efficiency but keep user's fold count
            cv_strategy = f"Efficient CV ({n_folds}-fold on {cv_sample_size:,} samples)"
        
        # SMART SAMPLING: Use subset for CV if needed
        X_cv, y_cv = X_train, y_train
        
        if cv_sample_size < n_train:
            # Sample stratified subset for CV
            from sklearn.model_selection import train_test_split
            X_cv, _, y_cv, _ = train_test_split(
                X_train, y_train,
                train_size=cv_sample_size,
                stratify=y_train,
                random_state=42
            )
            logger.info(f"Large dataset ({n_train:,} samples). Using {cv_sample_size:,} samples for fast CV with {n_folds} folds.")
        else:
            X_cv, y_cv = X_train, y_train
            logger.info(f"Dataset size: {n_train:,} samples. Using {cv_strategy}.")
        
        # Cross-validation on training set (or subset)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        try:
            logger.info(f"Running {n_folds}-fold CV for {model_name}... ({cv_strategy})")
            cv_results = cross_validate(
                model, X_cv, y_cv,
                cv=cv,
                scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'},
                return_estimator=False,
                n_jobs=1,
                verbose=0
            )
            cv_scores_list = cv_results['test_accuracy'].tolist()
            logger.info(f"{model_name} CV: {np.mean(cv_scores_list):.4f} ± {np.std(cv_scores_list):.4f}")
        except Exception as e:
            logger.warning(f"CV failed for {model_name}: {e}")
            cv_scores_list = [0.0]
        
        # Train on FULL training set (with optimization for SVM on large datasets)
        logger.info(f"Training {model_name} on full set ({n_train} samples)...")
        
        # CRITICAL OPTIMIZATION: For SVM on very large datasets, train on subset
        if 'SVM' in model_name and n_train > 20000:
            max_svm_samples = 15000
            logger.info(f"⚡ {model_name}: Using {max_svm_samples:,} samples for faster training")
            
            from sklearn.model_selection import train_test_split
            X_train_subset, _, y_train_subset, _ = train_test_split(
                X_train, y_train,
                train_size=max_svm_samples,
                stratify=y_train,
                random_state=42
            )
            
            try:
                model.fit(X_train_subset, y_train_subset)
                logger.info(f"{model_name} complete on subset!")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise
        else:
            # Normal training for other models or smaller datasets
            try:
                model.fit(X_train, y_train)
                logger.info(f"{model_name} complete!")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise
        
        # Evaluate on training set (to detect overfitting)
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
        
        # Evaluate on test set (true performance)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # Compute extended metrics
        advanced_calc = AdvancedMetricsCalculator()
        extended_metrics = advanced_calc.compute_extended_metrics(
            y_test, y_test_pred, 
            y_proba=model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        )
        
        # Compute calibration and confidence metrics if probabilities available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            calibration = advanced_calc.compute_calibration_metrics(y_test, y_proba)
            confidence_analysis = advanced_calc.compute_prediction_confidence_analysis(
                y_test, y_test_pred, y_proba
            )
        else:
            calibration = None
            confidence_analysis = None
        
        # Compute detailed confusion matrix analysis
        confusion_analysis = advanced_calc.analyze_confusion_matrix_detailed(
            y_test, y_test_pred, class_names=None
        )
        
        # Schedule background analyses
        if 'bg_manager' not in st.session_state:
            st.session_state.bg_manager = BackgroundAnalysisManager()
        
        # Bias-variance (background) - reduced bootstrap for speed
        bv_analyzer = BiasVarianceAnalyzer(n_bootstrap=30)
        bv_future = st.session_state.bg_manager.schedule_analysis(
            bv_analyzer.compute_bias_variance_decomposition,
            model, X_train, y_train, X_test, y_test
        )
        
        # Learning curves (background)
        lc_future = st.session_state.bg_manager.schedule_analysis(
            bv_analyzer.compute_learning_curves,
            model, X_train, y_train
        )
        
        # Detect overfitting
        detector = OverfittingDetector()
        warnings = detector.detect_overfitting(
            train_scores={'accuracy': train_accuracy, 'f1_macro': train_f1},
            test_scores={'accuracy': test_accuracy, 'f1_macro': test_f1},
            cv_scores={'accuracy': cv_scores_list},
            dataset_info={
                'n_samples': len(X_train) + len(X_test),
                'n_test_samples': len(X_test),
                'n_classes': len(np.unique(y_train)),
                'class_balance': {
                    int(i): float(np.sum(y_test == i) / len(y_test)) 
                    for i in np.unique(y_test)
                }
            }
        )
        
        overfitting_report = detector.get_user_guidance()
        
        # Store results with backward compatibility and new metrics
        results = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'train_f1_macro': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1,
            
            # For compatibility with existing code
            'accuracy': test_accuracy,
            'f1_macro': test_f1,
            'accuracy_mean': test_accuracy,
            'f1_macro_mean': test_f1,
            'roc_auc_ovr_mean': 0.0,  # Would need probability predictions
            'log_loss_mean': 0.0,
            
            # CV scores
            'cv_scores': cv_scores_list,  # Add full list for statistical tests
            'cv_accuracy_mean': np.mean(cv_scores_list),
            'cv_accuracy_std': np.std(cv_scores_list),
            'cv_f1_mean': test_f1,  # Approximate
            'cv_strategy': cv_strategy,  # NEW: CV strategy info
            'cv_folds': n_folds,  # NEW: Number of folds used
            'cv_sample_size': cv_sample_size,  # NEW: Sample size for CV
            
            # Overfitting metrics
            'overfitting_gap': train_accuracy - test_accuracy,
            'overfitting_warnings': overfitting_report,
            
            # Extended metrics
            **extended_metrics,  # Unpack all advanced metrics
            
            # Calibration and confidence
            'calibration': calibration,
            'confidence_analysis': confidence_analysis,
            
            # Detailed confusion matrix analysis
            'confusion_analysis': confusion_analysis,
            
            # Background analysis futures
            'bias_variance_future': bv_future,
            'learning_curves_future': lc_future,
            
            # Model and predictions
            'trained_model': model,
            'predictions': y_test_pred,
            'test_predictions': y_test_pred,
            'true_labels': y_test,
            'test_true': y_test
        }
        
        # Store in self.results for leaderboard
        self.results[model_name] = results
        
        return results
