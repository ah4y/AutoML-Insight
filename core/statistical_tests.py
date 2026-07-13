"""Statistical Significance Testing for Model Comparison.

This module provides statistical tests to determine if performance differences
between models are statistically significant.
"""

import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalModelComparator:
    """Perform statistical significance tests for model comparison.
    
    This class provides methods to compare multiple models using statistical
    tests like paired t-test, Wilcoxon signed-rank test, and Friedman test.
    """
    
    def __init__(self):
        """Initialize the StatisticalModelComparator."""
        logger.info("StatisticalModelComparator initialized")
    
    def compute_pairwise_tests(self, results_dict):
        """Compute pairwise statistical tests between models.
        
        Performs both parametric (paired t-test) and non-parametric (Wilcoxon)
        tests to compare cross-validation scores between pairs of models.
        
        Args:
            results_dict: Dictionary mapping model names to their results dicts
                         Each result dict should contain 'cv_scores' key
            
        Returns:
            Dictionary with pairwise test results:
                - pairwise_ttest: Dict of (model1, model2) -> p-value
                - pairwise_wilcoxon: Dict of (model1, model2) -> p-value
                - significant_differences: List of significant differences
        """
        try:
            logger.info("Computing pairwise statistical tests")
            
            # Extract model names and CV scores
            models_with_cv = {}
            for model_name, results in results_dict.items():
                if 'cv_scores' in results and results['cv_scores'] is not None:
                    cv_scores = results['cv_scores']
                    if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 1:
                        models_with_cv[model_name] = np.array(cv_scores)
            
            if len(models_with_cv) < 2:
                logger.warning("Need at least 2 models with CV scores for pairwise tests")
                return None
            
            model_names = list(models_with_cv.keys())
            pairwise_ttest = {}
            pairwise_wilcoxon = {}
            significant_differences = []
            
            # Compare all pairs
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Only compute upper triangle
                        scores1 = models_with_cv[model1]
                        scores2 = models_with_cv[model2]
                        
                        # Ensure equal length
                        min_len = min(len(scores1), len(scores2))
                        scores1 = scores1[:min_len]
                        scores2 = scores2[:min_len]
                        
                        # Paired t-test
                        try:
                            t_stat, t_pval = stats.ttest_rel(scores1, scores2)
                            pairwise_ttest[(model1, model2)] = float(t_pval)
                        except Exception as e:
                            logger.warning(f"T-test failed for {model1} vs {model2}: {e}")
                            pairwise_ttest[(model1, model2)] = None
                        
                        # Wilcoxon signed-rank test
                        try:
                            w_stat, w_pval = stats.wilcoxon(scores1, scores2)
                            pairwise_wilcoxon[(model1, model2)] = float(w_pval)
                        except Exception as e:
                            logger.warning(f"Wilcoxon test failed for {model1} vs {model2}: {e}")
                            pairwise_wilcoxon[(model1, model2)] = None
                        
                        # Check for significant difference (p < 0.05)
                        if t_pval is not None and t_pval < 0.05:
                            winner = model1 if scores1.mean() > scores2.mean() else model2
                            loser = model2 if winner == model1 else model1
                            significant_differences.append({
                                'winner': winner,
                                'loser': loser,
                                'p_value': float(t_pval),
                                'test': 't-test',
                                'interpretation': self.interpret_pvalue(t_pval)
                            })
            
            logger.info(f"Computed {len(pairwise_ttest)} pairwise comparisons")
            logger.info(f"Found {len(significant_differences)} significant differences")
            
            return {
                'pairwise_ttest': pairwise_ttest,
                'pairwise_wilcoxon': pairwise_wilcoxon,
                'significant_differences': significant_differences
            }
            
        except Exception as e:
            logger.error(f"Error computing pairwise tests: {e}")
            return None
    
    def compute_friedman_test(self, results_dict):
        """Compute Friedman test for multiple model comparison.
        
        The Friedman test is a non-parametric alternative to repeated measures
        ANOVA, used to detect differences in treatments across multiple test attempts.
        
        Args:
            results_dict: Dictionary mapping model names to their results dicts
                         Each result dict should contain 'cv_scores' key
            
        Returns:
            Dictionary with Friedman test results:
                - statistic: Test statistic
                - p_value: P-value
                - interpretation: Human-readable interpretation
                - model_rankings: Mean rank of each model
        """
        try:
            logger.info("Computing Friedman test")
            
            # Extract CV scores for models
            models_with_cv = {}
            for model_name, results in results_dict.items():
                if 'cv_scores' in results and results['cv_scores'] is not None:
                    cv_scores = results['cv_scores']
                    if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 1:
                        models_with_cv[model_name] = np.array(cv_scores)
            
            if len(models_with_cv) < 3:
                logger.warning("Need at least 3 models for Friedman test")
                return None
            
            # Prepare data for Friedman test
            model_names = list(models_with_cv.keys())
            scores_arrays = [models_with_cv[name] for name in model_names]
            
            # Ensure all arrays have same length
            min_len = min(len(arr) for arr in scores_arrays)
            scores_arrays = [arr[:min_len] for arr in scores_arrays]
            
            # Perform Friedman test
            statistic, p_value = stats.friedmanchisquare(*scores_arrays)
            
            # Compute rankings
            # Stack scores and compute ranks across models for each CV fold
            stacked = np.column_stack(scores_arrays)
            ranks = np.apply_along_axis(lambda x: stats.rankdata(-x), axis=1, arr=stacked)
            mean_ranks = ranks.mean(axis=0)
            
            model_rankings = {
                model_names[i]: float(mean_ranks[i])
                for i in range(len(model_names))
            }
            
            # Sort by rank (lower is better)
            sorted_rankings = sorted(model_rankings.items(), key=lambda x: x[1])
            
            interpretation = self.interpret_pvalue(p_value)
            if p_value < 0.05:
                interpretation += f" - Models have significantly different performance. Best: {sorted_rankings[0][0]}"
            else:
                interpretation += " - No significant difference between models"
            
            logger.info(f"Friedman test: statistic={statistic:.4f}, p={p_value:.4f}")
            logger.info(f"Best model by rank: {sorted_rankings[0][0]} (rank={sorted_rankings[0][1]:.2f})")
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'interpretation': interpretation,
                'model_rankings': model_rankings,
                'sorted_rankings': sorted_rankings
            }
            
        except Exception as e:
            logger.error(f"Error computing Friedman test: {e}")
            return None
    
    def compute_cv_stability(self, cv_scores_list):
        """Compute stability metrics for cross-validation scores.
        
        Stability indicates how consistent a model's performance is across
        different data splits. High stability (low variance) suggests reliable performance.
        
        Args:
            cv_scores_list: List or array of cross-validation scores
            
        Returns:
            Dictionary with stability metrics:
                - mean: Mean CV score
                - std: Standard deviation
                - stability_index: 1 - coefficient_of_variation (higher is better)
                - coefficient_of_variation: std/mean (lower is better)
                - interpretation: Human-readable interpretation
        """
        try:
            if cv_scores_list is None or len(cv_scores_list) < 2:
                return None
            
            scores = np.array(cv_scores_list)
            mean_score = float(scores.mean())
            std_score = float(scores.std())
            
            # Coefficient of variation (CV)
            cv = std_score / mean_score if mean_score > 0 else float('inf')
            
            # Stability index (higher is better, range 0-1)
            stability_index = 1.0 - min(cv, 1.0)
            
            # Interpretation
            if stability_index > 0.9:
                interpretation = "Excellent stability - Very consistent performance"
            elif stability_index > 0.8:
                interpretation = "Good stability - Reliable performance"
            elif stability_index > 0.7:
                interpretation = "Moderate stability - Some variability"
            else:
                interpretation = "Low stability - High variability across folds"
            
            logger.info(f"CV stability: mean={mean_score:.4f}, std={std_score:.4f}, "
                       f"stability_index={stability_index:.4f}")
            
            return {
                'mean': mean_score,
                'std': std_score,
                'stability_index': float(stability_index),
                'coefficient_of_variation': float(cv),
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Error computing CV stability: {e}")
            return None
    
    @staticmethod
    def interpret_pvalue(pvalue):
        """Interpret p-value with human-readable text.
        
        Args:
            pvalue: P-value from statistical test
            
        Returns:
            Human-readable interpretation string
        """
        if pvalue is None:
            return "Test could not be performed"
        elif pvalue < 0.001:
            return "Highly significant difference (p < 0.001)"
        elif pvalue < 0.01:
            return "Very significant difference (p < 0.01)"
        elif pvalue < 0.05:
            return "Significant difference (p < 0.05)"
        elif pvalue < 0.1:
            return "Marginally significant (p < 0.1)"
        else:
            return "No significant difference (p ≥ 0.1)"
