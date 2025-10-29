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
                except:
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
        except:
            pass
        
        # Wilcoxon test (for scores)
        if 'accuracy_scores' in results1 and 'accuracy_scores' in results2:
            try:
                wilcoxon_p = wilcoxon_test(
                    np.array(results1['accuracy_scores']),
                    np.array(results2['accuracy_scores'])
                )
                comparison['wilcoxon_p_value'] = wilcoxon_p
            except:
                pass
        
        return comparison
    
    def get_leaderboard(self, metric: str = 'accuracy') -> List[Dict[str, Any]]:
        """
        Get model leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by
            
        Returns:
            Sorted list of model results
        """
        leaderboard = []
        
        for model_name, results in self.results.items():
            metric_key = f'{metric}_mean'
            if metric_key in results:
                leaderboard.append({
                    'model': model_name,
                    'score': results[metric_key],
                    'ci_lower': results.get(f'{metric}_ci_lower', 0),
                    'ci_upper': results.get(f'{metric}_ci_upper', 0)
                })
        
        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        
        return leaderboard
