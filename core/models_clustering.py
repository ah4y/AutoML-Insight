"""Clustering models."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


class AutoKMeans:
    """KMeans with automatic k selection using elbow and silhouette methods."""

    def __init__(self, k_range: tuple = (2, 10), random_state: int = 42, max_samples: int = None):
        self.k_range = k_range
        self.random_state = random_state
        self.max_samples = max_samples or 10000  # Default limit for k-selection
        self.best_k = None
        self.model = None
        self.inertias = []
        self.silhouette_scores = []

    def fit(self, X):
        """Fit KMeans with automatic k selection."""
        # Performance optimization: Use sampling for k-selection on large datasets
        X_for_k_selection = X
        if len(X) > self.max_samples:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X_for_k_selection = X[indices]

        best_score = -1

        for k in range(self.k_range[0], self.k_range[1] + 1):
            # Use faster initialization and fewer iterations for k-selection
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=5,  # Reduced from 10
                max_iter=100,  # Reduced from 300
                algorithm="elkan",  # Faster for dense data
            )
            labels = kmeans.fit_predict(X_for_k_selection)

            self.inertias.append(kmeans.inertia_)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_for_k_selection, labels)
                self.silhouette_scores.append(score)

                if score > best_score:
                    best_score = score
                    self.best_k = k
            else:
                self.silhouette_scores.append(-1)

        if self.best_k is None:
            self.best_k = min(3, self.k_range[1])  # Safe default

        # Now fit final model on full dataset with best k
        self.model = KMeans(
            n_clusters=self.best_k,
            random_state=self.random_state,
            n_init=10,  # Full initialization for final model
            algorithm="elkan",
        )
        self.model.fit(X)

        return self

    def predict(self, X):
        """Predict cluster labels."""
        return self.model.predict(X)

    def fit_predict(self, X):
        """Fit and predict."""
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)."""
        return {"k_range": self.k_range, "random_state": self.random_state}

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AutoGMM:
    """Gaussian Mixture Model with automatic component selection using BIC/AIC."""

    def __init__(self, k_range: tuple = (2, 10), random_state: int = 42, max_samples: int = None):
        self.k_range = k_range
        self.random_state = random_state
        self.max_samples = max_samples or 10000  # Default limit for component selection
        self.best_k = None
        self.model = None
        self.bic_scores = []
        self.aic_scores = []

    def fit(self, X):
        """Fit GMM with automatic component selection."""
        # Performance optimization: Use sampling for component selection on large datasets
        X_for_k_selection = X
        if len(X) > self.max_samples:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X_for_k_selection = X[indices]

        best_bic = float("inf")

        for k in range(self.k_range[0], self.k_range[1] + 1):
            gmm = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
                covariance_type="diag",  # Faster than 'full' for large datasets
                max_iter=50,  # Reduced from default 100
            )
            try:
                gmm.fit(X_for_k_selection)

                bic = gmm.bic(X_for_k_selection)
                aic = gmm.aic(X_for_k_selection)

                self.bic_scores.append(bic)
                self.aic_scores.append(aic)

                if bic < best_bic:
                    best_bic = bic
                    self.best_k = k
            except:
                # Handle convergence failures
                self.bic_scores.append(float("inf"))
                self.aic_scores.append(float("inf"))

        # Fit final model on full dataset
        if self.best_k is None:
            self.best_k = min(3, self.k_range[1])  # Safe default

        self.model = GaussianMixture(
            n_components=self.best_k, random_state=self.random_state, covariance_type="diag", max_iter=100
        )
        self.model.fit(X)

        return self

    def predict(self, X):
        """Predict cluster labels."""
        return self.model.predict(X)

    def fit_predict(self, X):
        """Fit and predict."""
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)."""
        return {"k_range": self.k_range, "random_state": self.random_state}

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AutoDBSCAN:
    """DBSCAN with automatic epsilon selection."""

    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self.eps = None
        self.model = None
        self.X_train = None

    def fit(self, X):
        """Fit DBSCAN with automatic epsilon selection."""
        # Store training data for predict method
        self.X_train = X.copy() if hasattr(X, "copy") else np.array(X)

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

    def predict(self, X):
        """Predict cluster labels for new data using DBSCAN logic.

        Uses a nearest-neighbor approach: assigns to the cluster of the
        nearest training point if within eps distance, otherwise marks as noise (-1).
        """
        if self.model is None or self.X_train is None:
            raise ValueError("Model must be fitted before calling predict()")

        from sklearn.metrics.pairwise import pairwise_distances

        # Compute distances from X to all training points
        distances = pairwise_distances(X, self.X_train)

        # For each test point, find the nearest training point
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(X)), nearest_indices]

        # Get the labels of nearest training points
        labels = self.model.labels_[nearest_indices].copy()

        # Mark as noise if the nearest point is further than eps
        labels[nearest_distances > self.eps] = -1

        return labels

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)."""
        return {"min_samples": self.min_samples}

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def get_clustering_models(random_state: int = 42, n_samples: int = None) -> Dict[str, Any]:
    """
    Get a dictionary of clustering models optimized for dataset size.

    Args:
        random_state: Random seed for reproducibility
        n_samples: Number of samples in dataset for optimization

    Returns:
        Dictionary mapping model names to model instances
    """
    # Determine if we have a large dataset
    is_large_dataset = n_samples is not None and n_samples > 10000

    models = {
        "KMeans": AutoKMeans(
            k_range=(2, 8 if is_large_dataset else 10),
            random_state=random_state,
            max_samples=min(10000, n_samples) if n_samples else 10000,
        ),
        "GMM": AutoGMM(
            k_range=(2, 6 if is_large_dataset else 10),
            random_state=random_state,
            max_samples=min(10000, n_samples) if n_samples else 10000,
        ),
    }

    # Only add computationally expensive models for smaller datasets
    if not is_large_dataset:
        models.update(
            {
                "DBSCAN": AutoDBSCAN(min_samples=5),
                "Agglomerative": AgglomerativeClustering(n_clusters=3),
                "Spectral": SpectralClustering(n_clusters=3, random_state=random_state, assign_labels="kmeans"),
            }
        )

    return models
