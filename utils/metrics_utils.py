"""Metrics computation utilities."""

import numpy as np
from scipy import stats
from typing import Tuple, List


def compute_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for scores.
    
    Args:
        scores: List of metric scores
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(scores)
    if len(scores) == 1:
        return mean, mean, mean
    
    std_err = stats.sem(scores)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    
    return mean, mean - margin, mean + margin


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        scores: Array of metric scores
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    
    means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    
    return np.mean(scores), lower, upper


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Perform McNemar's test for paired predictions.
    
    Args:
        y_true: True labels
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        
    Returns:
        p-value
    """
    # Contingency table
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    
    n01 = np.sum(~correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    
    # McNemar's test with continuity correction
    if n01 + n10 == 0:
        return 1.0
    
    statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return p_value


def wilcoxon_test(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        p-value
    """
    try:
        statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
        return p_value
    except ValueError:
        return 1.0
