"""Clustering evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import ShuffleSplit
from typing import Dict, Any


class ClusteringEvaluator:
    """Comprehensive evaluation for clustering models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str,
        labels: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate a clustering model.
        
        Args:
            model: Fitted clustering model
            X: Feature matrix
            model_name: Name of the model
            labels: Cluster labels (if already predicted)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get cluster labels
        if labels is None:
            try:
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            except:
                labels = model.fit_predict(X)
        
        results = {
            'model_name': model_name,
            'n_clusters': len(np.unique(labels[labels >= 0])),  # Exclude noise (-1)
            'labels': labels
        }
        
        # Only compute metrics if we have valid clusters
        valid_labels = labels[labels >= 0]
        if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 0:
            X_valid = X[labels >= 0]
            
            try:
                # Silhouette score
                results['silhouette'] = silhouette_score(X_valid, valid_labels)
            except:
                results['silhouette'] = -1
            
            try:
                # Davies-Bouldin index (lower is better)
                results['davies_bouldin'] = davies_bouldin_score(X_valid, valid_labels)
            except:
                results['davies_bouldin'] = float('inf')
            
            try:
                # Calinski-Harabasz index (higher is better)
                results['calinski_harabasz'] = calinski_harabasz_score(X_valid, valid_labels)
            except:
                results['calinski_harabasz'] = 0
            
            # Cluster stability
            results['stability'] = self._compute_stability(model, X)
        else:
            results['silhouette'] = -1
            results['davies_bouldin'] = float('inf')
            results['calinski_harabasz'] = 0
            results['stability'] = 0
        
        # Noise ratio (for DBSCAN)
        results['noise_ratio'] = np.sum(labels == -1) / len(labels)
        
        self.results[model_name] = results
        return results
    
    def _compute_stability(
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 10
    ) -> float:
        """
        Compute cluster stability using bootstrap resampling.
        
        Args:
            model: Clustering model
            X: Feature matrix
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Stability score (0-1)
        """
        from sklearn.base import clone
        from sklearn.metrics import adjusted_rand_score
        
        base_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        stability_scores = []
        
        splitter = ShuffleSplit(
            n_splits=n_iterations,
            test_size=0.3,
            random_state=self.random_state
        )
        
        for train_idx, test_idx in splitter.split(X):
            try:
                X_train = X[train_idx]
                
                # Clone and fit on bootstrap sample
                model_clone = clone(model)
                if hasattr(model_clone, 'fit_predict'):
                    new_labels = model_clone.fit_predict(X_train)
                else:
                    model_clone.fit(X_train)
                    new_labels = model_clone.predict(X_train)
                
                # Compare with base clustering
                base_labels_train = base_labels[train_idx]
                
                # Only compare valid labels
                valid_mask = (new_labels >= 0) & (base_labels_train >= 0)
                if np.sum(valid_mask) > 0:
                    ari = adjusted_rand_score(
                        base_labels_train[valid_mask],
                        new_labels[valid_mask]
                    )
                    stability_scores.append(max(0, ari))
            except:
                continue
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def get_leaderboard(self, metric: str = 'silhouette') -> list:
        """
        Get clustering model leaderboard.
        
        Args:
            metric: Metric to sort by
            
        Returns:
            Sorted list of model results
        """
        leaderboard = []
        
        for model_name, results in self.results.items():
            if metric in results:
                score = results[metric]
                # Handle inf values
                if np.isinf(score):
                    score = -999 if metric == 'davies_bouldin' else 999
                
                leaderboard.append({
                    'model': model_name,
                    'score': score,
                    'n_clusters': results['n_clusters']
                })
        
        # Sort (ascending for davies_bouldin, descending for others)
        reverse = (metric != 'davies_bouldin')
        leaderboard.sort(key=lambda x: x['score'], reverse=reverse)
        
        return leaderboard
