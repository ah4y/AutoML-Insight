"""Clustering models."""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Tuple


class AutoKMeans:
    """KMeans with automatic k selection using elbow and silhouette methods."""
    
    def __init__(self, k_range: tuple = (2, 10), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.best_k = None
        self.model = None
        self.inertias = []
        self.silhouette_scores = []
    
    def fit(self, X):
        """Fit KMeans with automatic k selection."""
        best_score = -1
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            self.inertias.append(kmeans.inertia_)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                self.silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    self.best_k = k
                    self.model = kmeans
            else:
                self.silhouette_scores.append(-1)
        
        if self.model is None:
            self.best_k = 2
            self.model = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            self.model.fit(X)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels."""
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """Fit and predict."""
        self.fit(X)
        return self.predict(X)


class AutoGMM:
    """Gaussian Mixture Model with automatic component selection using BIC/AIC."""
    
    def __init__(self, k_range: tuple = (2, 10), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.best_k = None
        self.model = None
        self.bic_scores = []
        self.aic_scores = []
    
    def fit(self, X):
        """Fit GMM with automatic component selection."""
        best_bic = float('inf')
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            gmm = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
                covariance_type='full'
            )
            gmm.fit(X)
            
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            self.bic_scores.append(bic)
            self.aic_scores.append(aic)
            
            if bic < best_bic:
                best_bic = bic
                self.best_k = k
                self.model = gmm
        
        return self
    
    def predict(self, X):
        """Predict cluster labels."""
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """Fit and predict."""
        self.fit(X)
        return self.predict(X)


class AutoDBSCAN:
    """DBSCAN with automatic epsilon selection."""
    
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self.eps = None
        self.model = None
    
    def fit(self, X):
        """Fit DBSCAN with automatic epsilon selection."""
        # Estimate eps using k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        
        # Use 95th percentile of k-nearest neighbor distances
        distances = np.sort(distances[:, -1])
        self.eps = np.percentile(distances, 95)
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.model.fit(X)
        
        return self
    
    def fit_predict(self, X):
        """Fit and predict."""
        self.fit(X)
        return self.model.labels_


def get_clustering_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Get a dictionary of clustering models.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'KMeans': AutoKMeans(k_range=(2, 10), random_state=random_state),
        'GMM': AutoGMM(k_range=(2, 10), random_state=random_state),
        'DBSCAN': AutoDBSCAN(min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=3),
        'Spectral': SpectralClustering(
            n_clusters=3,
            random_state=random_state,
            assign_labels='kmeans'
        )
    }
    
    return models
