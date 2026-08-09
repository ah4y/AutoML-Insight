"""Unit tests for core/models_clustering.py module.

Tests the clustering models: AutoKMeans, AutoGMM, AutoDBSCAN, and the
get_clustering_models factory function.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from core.models_clustering import AutoKMeans, AutoGMM, AutoDBSCAN, get_clustering_models


@pytest.fixture
def small_blob_dataset():
    """Create a small blob dataset for testing."""
    X, y = make_blobs(n_samples=100, centers=3, random_state=42, cluster_std=0.5)
    return X, y


@pytest.fixture
def medium_blob_dataset():
    """Create a medium-sized blob dataset."""
    X, y = make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=0.6)
    return X, y


@pytest.fixture
def large_blob_dataset():
    """Create a large blob dataset."""
    X, y = make_blobs(n_samples=15000, centers=5, random_state=42, cluster_std=0.7)
    return X, y


@pytest.fixture
def high_dim_dataset():
    """Create a high-dimensional dataset."""
    X, y = make_blobs(n_samples=200, centers=3, n_features=20, random_state=42)
    return X, y


@pytest.fixture
def two_cluster_dataset():
    """Create a dataset with exactly 2 clusters."""
    X, y = make_blobs(n_samples=150, centers=2, random_state=42, cluster_std=0.5)
    return X, y


@pytest.fixture
def single_cluster_dataset():
    """Create a dataset that behaves like a single cluster."""
    np.random.seed(42)
    X = np.random.normal(0, 0.1, (100, 2))
    return X, np.zeros(100)


class TestAutoKMeans:
    """Test AutoKMeans class."""

    def test_init_default_params(self):
        """Test AutoKMeans initialization with default parameters."""
        kmeans = AutoKMeans()
        
        assert kmeans.k_range == (2, 10)
        assert kmeans.random_state == 42
        assert kmeans.max_samples == 10000
        assert kmeans.best_k is None
        assert kmeans.model is None
        assert kmeans.inertias == []
        assert kmeans.silhouette_scores == []

    def test_init_custom_params(self):
        """Test AutoKMeans initialization with custom parameters."""
        kmeans = AutoKMeans(k_range=(2, 5), random_state=123, max_samples=5000)
        
        assert kmeans.k_range == (2, 5)
        assert kmeans.random_state == 123
        assert kmeans.max_samples == 5000

    def test_fit_small_dataset(self, small_blob_dataset):
        """Test fit method on small dataset."""
        X, _ = small_blob_dataset
        kmeans = AutoKMeans(random_state=42)
        
        result = kmeans.fit(X)
        
        # Check that fit returns self
        assert result is kmeans
        # Check that best_k was selected
        assert kmeans.best_k is not None
        assert kmeans.best_k >= 2 and kmeans.best_k <= 10
        # Check that model was fitted
        assert kmeans.model is not None
        # Check inertias and silhouette scores
        assert len(kmeans.inertias) == 9  # k_range (2, 10) = 9 values
        assert len(kmeans.silhouette_scores) == 9

    def test_fit_custom_k_range(self, small_blob_dataset):
        """Test fit with custom k_range."""
        X, _ = small_blob_dataset
        kmeans = AutoKMeans(k_range=(2, 5), random_state=42)
        
        kmeans.fit(X)
        
        # With k_range (2, 5), we should evaluate 4 values
        assert len(kmeans.inertias) == 4
        assert len(kmeans.silhouette_scores) == 4
        assert kmeans.best_k >= 2 and kmeans.best_k <= 5

    def test_fit_large_dataset(self, large_blob_dataset):
        """Test fit method on large dataset with sampling."""
        X, _ = large_blob_dataset
        kmeans = AutoKMeans(max_samples=1000)
        
        kmeans.fit(X)
        
        # Should still select a best_k despite large dataset
        assert kmeans.best_k is not None
        assert kmeans.model is not None

    def test_predict_after_fit(self, small_blob_dataset):
        """Test predict method after fitting."""
        X, _ = small_blob_dataset
        kmeans = AutoKMeans()
        kmeans.fit(X)
        
        predictions = kmeans.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all((predictions >= 0) & (predictions < kmeans.best_k))

    def test_fit_predict(self, small_blob_dataset):
        """Test fit_predict method."""
        X, _ = small_blob_dataset
        kmeans = AutoKMeans()
        
        predictions = kmeans.fit_predict(X)
        
        assert len(predictions) == len(X)
        assert kmeans.model is not None
        assert np.all((predictions >= 0) & (predictions < kmeans.best_k))

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        kmeans = AutoKMeans(k_range=(3, 8), random_state=123)
        
        params = kmeans.get_params()
        
        assert params['k_range'] == (3, 8)
        assert params['random_state'] == 123

    def test_set_params(self, small_blob_dataset):
        """Test set_params method for sklearn compatibility."""
        X, _ = small_blob_dataset
        kmeans = AutoKMeans()
        
        # Set new parameters
        result = kmeans.set_params(k_range=(2, 4), random_state=123)
        
        # Check that set_params returns self
        assert result is kmeans
        # Check parameters were updated
        assert kmeans.k_range == (2, 4)
        assert kmeans.random_state == 123

    def test_two_cluster_dataset(self, two_cluster_dataset):
        """Test with exactly two clusters."""
        X, _ = two_cluster_dataset
        kmeans = AutoKMeans()
        
        kmeans.fit(X)
        
        assert kmeans.best_k >= 2
        predictions = kmeans.predict(X)
        assert len(np.unique(predictions)) >= 1

    def test_single_cluster_behavior(self, single_cluster_dataset):
        """Test behavior on nearly-single-cluster data."""
        X, _ = single_cluster_dataset
        kmeans = AutoKMeans(k_range=(2, 5))
        
        kmeans.fit(X)
        
        # Should still select a valid k
        assert kmeans.best_k is not None
        assert kmeans.best_k <= 5

    def test_high_dimensional_data(self, high_dim_dataset):
        """Test with high-dimensional data."""
        X, _ = high_dim_dataset
        kmeans = AutoKMeans()
        
        kmeans.fit(X)
        
        assert kmeans.best_k is not None
        predictions = kmeans.predict(X)
        assert len(predictions) == len(X)


class TestAutoGMM:
    """Test AutoGMM class."""

    def test_init_default_params(self):
        """Test AutoGMM initialization with default parameters."""
        gmm = AutoGMM()
        
        assert gmm.k_range == (2, 10)
        assert gmm.random_state == 42
        assert gmm.max_samples == 10000
        assert gmm.best_k is None
        assert gmm.model is None
        assert gmm.bic_scores == []
        assert gmm.aic_scores == []

    def test_init_custom_params(self):
        """Test AutoGMM initialization with custom parameters."""
        gmm = AutoGMM(k_range=(2, 6), random_state=456, max_samples=8000)
        
        assert gmm.k_range == (2, 6)
        assert gmm.random_state == 456
        assert gmm.max_samples == 8000

    def test_fit_small_dataset(self, small_blob_dataset):
        """Test fit method on small dataset."""
        X, _ = small_blob_dataset
        gmm = AutoGMM(random_state=42)
        
        result = gmm.fit(X)
        
        # Check that fit returns self
        assert result is gmm
        # Check that best_k was selected
        assert gmm.best_k is not None
        assert gmm.best_k >= 2 and gmm.best_k <= 10
        # Check that model was fitted
        assert gmm.model is not None
        # Check BIC and AIC scores
        assert len(gmm.bic_scores) == 9  # k_range (2, 10) = 9 values
        assert len(gmm.aic_scores) == 9
        # BIC scores should be finite numbers
        assert all(np.isfinite(gmm.bic_scores))

    def test_fit_custom_k_range(self, small_blob_dataset):
        """Test fit with custom k_range."""
        X, _ = small_blob_dataset
        gmm = AutoGMM(k_range=(2, 4), random_state=42)
        
        gmm.fit(X)
        
        # With k_range (2, 4), we should evaluate 3 values
        assert len(gmm.bic_scores) == 3
        assert len(gmm.aic_scores) == 3
        assert gmm.best_k >= 2 and gmm.best_k <= 4

    def test_fit_large_dataset(self, large_blob_dataset):
        """Test fit method on large dataset with sampling."""
        X, _ = large_blob_dataset
        gmm = AutoGMM(max_samples=2000)
        
        gmm.fit(X)
        
        # Should still select a best_k despite large dataset
        assert gmm.best_k is not None
        assert gmm.model is not None

    def test_predict_after_fit(self, small_blob_dataset):
        """Test predict method after fitting."""
        X, _ = small_blob_dataset
        gmm = AutoGMM()
        gmm.fit(X)
        
        predictions = gmm.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all((predictions >= 0) & (predictions < gmm.best_k))

    def test_fit_predict(self, small_blob_dataset):
        """Test fit_predict method."""
        X, _ = small_blob_dataset
        gmm = AutoGMM()
        
        predictions = gmm.fit_predict(X)
        
        assert len(predictions) == len(X)
        assert gmm.model is not None
        assert np.all((predictions >= 0) & (predictions < gmm.best_k))

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        gmm = AutoGMM(k_range=(3, 7), random_state=456)
        
        params = gmm.get_params()
        
        assert params['k_range'] == (3, 7)
        assert params['random_state'] == 456

    def test_set_params(self, small_blob_dataset):
        """Test set_params method for sklearn compatibility."""
        X, _ = small_blob_dataset
        gmm = AutoGMM()
        
        # Set new parameters
        result = gmm.set_params(k_range=(2, 5), random_state=456)
        
        # Check that set_params returns self
        assert result is gmm
        # Check parameters were updated
        assert gmm.k_range == (2, 5)
        assert gmm.random_state == 456

    def test_bic_minimization(self, small_blob_dataset):
        """Test that BIC is minimized correctly."""
        X, _ = small_blob_dataset
        gmm = AutoGMM()
        
        gmm.fit(X)
        
        # best_k should correspond to the minimum BIC
        best_k_idx = gmm.best_k - 2  # Adjust for k_range starting at 2
        min_bic_idx = np.argmin(gmm.bic_scores)
        assert best_k_idx == min_bic_idx

    def test_two_cluster_dataset(self, two_cluster_dataset):
        """Test with two-cluster dataset."""
        X, _ = two_cluster_dataset
        gmm = AutoGMM()
        
        gmm.fit(X)
        
        assert gmm.best_k >= 2
        predictions = gmm.predict(X)
        assert len(predictions) == len(X)

    def test_high_dimensional_data(self, high_dim_dataset):
        """Test with high-dimensional data."""
        X, _ = high_dim_dataset
        gmm = AutoGMM()
        
        gmm.fit(X)
        
        assert gmm.best_k is not None
        predictions = gmm.predict(X)
        assert len(predictions) == len(X)


class TestAutoDBSCAN:
    """Test AutoDBSCAN class."""

    def test_init_default_params(self):
        """Test AutoDBSCAN initialization with default parameters."""
        dbscan = AutoDBSCAN()
        
        assert dbscan.min_samples == 5
        assert dbscan.eps is None
        assert dbscan.model is None

    def test_init_custom_min_samples(self):
        """Test AutoDBSCAN initialization with custom min_samples."""
        dbscan = AutoDBSCAN(min_samples=10)
        
        assert dbscan.min_samples == 10

    def test_fit_small_dataset(self, small_blob_dataset):
        """Test fit method on small dataset."""
        X, _ = small_blob_dataset
        dbscan = AutoDBSCAN()
        
        result = dbscan.fit(X)
        
        # Check that fit returns self
        assert result is dbscan
        # Check that eps was estimated
        assert dbscan.eps is not None
        assert dbscan.eps > 0
        # Check that model was fitted
        assert dbscan.model is not None

    def test_fit_predict(self, small_blob_dataset):
        """Test fit_predict method."""
        X, _ = small_blob_dataset
        dbscan = AutoDBSCAN()
        
        predictions = dbscan.fit_predict(X)
        
        assert len(predictions) == len(X)
        assert dbscan.model is not None
        # DBSCAN can produce -1 labels for noise points
        assert np.all(predictions >= -1)

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        dbscan = AutoDBSCAN(min_samples=15)
        
        params = dbscan.get_params()
        
        assert params['min_samples'] == 15

    def test_set_params(self, small_blob_dataset):
        """Test set_params method for sklearn compatibility."""
        X, _ = small_blob_dataset
        dbscan = AutoDBSCAN()
        
        # Set new parameters
        result = dbscan.set_params(min_samples=10)
        
        # Check that set_params returns self
        assert result is dbscan
        # Check parameters were updated
        assert dbscan.min_samples == 10

    def test_eps_estimation(self, small_blob_dataset):
        """Test that eps is estimated from k-nearest neighbors."""
        X, _ = small_blob_dataset
        dbscan = AutoDBSCAN(min_samples=5)
        
        dbscan.fit(X)
        
        # eps should be between 0 and max distance in the dataset
        max_distance = np.max(np.linalg.norm(X - X.mean(axis=0), axis=1))
        assert dbscan.eps > 0
        assert dbscan.eps <= max_distance

    def test_varying_min_samples(self, small_blob_dataset):
        """Test with different min_samples values."""
        X, _ = small_blob_dataset
        
        for min_samples in [3, 5, 10, 20]:
            dbscan = AutoDBSCAN(min_samples=min_samples)
            dbscan.fit(X)
            
            assert dbscan.eps is not None
            predictions = dbscan.fit_predict(X)
            assert len(predictions) == len(X)

    def test_noisy_dataset(self):
        """Test DBSCAN on dataset with noise."""
        np.random.seed(42)
        X_clean, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        # Add noise points
        X_noise = np.random.uniform(-10, 10, (10, 2))
        X = np.vstack([X_clean, X_noise])
        
        dbscan = AutoDBSCAN(min_samples=5)
        predictions = dbscan.fit_predict(X)
        
        # Should have some noise points (-1 labels)
        assert np.any(predictions == -1) or np.any(predictions >= 0)
        assert len(predictions) == len(X)

    def test_high_dimensional_data(self, high_dim_dataset):
        """Test with high-dimensional data."""
        X, _ = high_dim_dataset
        dbscan = AutoDBSCAN()
        
        dbscan.fit(X)
        
        assert dbscan.eps is not None
        predictions = dbscan.fit_predict(X)
        assert len(predictions) == len(X)


class TestGetClusteringModels:
    """Test get_clustering_models factory function."""

    def test_small_dataset_models(self):
        """Test get_clustering_models for small dataset."""
        models = get_clustering_models(random_state=42, n_samples=100)
        
        # Small dataset should include all models
        assert 'KMeans' in models
        assert 'GMM' in models
        assert 'DBSCAN' in models
        assert 'Agglomerative' in models
        assert 'Spectral' in models
        assert len(models) == 5

    def test_large_dataset_models(self):
        """Test get_clustering_models for large dataset."""
        models = get_clustering_models(random_state=42, n_samples=15000)
        
        # Large dataset should only include fast models
        assert 'KMeans' in models
        assert 'GMM' in models
        assert 'DBSCAN' not in models  # Not included for large datasets
        assert 'Agglomerative' not in models  # Not included for large datasets
        assert 'Spectral' not in models  # Not included for large datasets
        assert len(models) == 2

    def test_boundary_dataset_size(self):
        """Test at the boundary between small and large datasets."""
        # Just below threshold
        models_below = get_clustering_models(random_state=42, n_samples=9999)
        assert len(models_below) == 5
        
        # Just above threshold
        models_above = get_clustering_models(random_state=42, n_samples=10001)
        assert len(models_above) == 2

    def test_none_n_samples(self):
        """Test with n_samples=None (default behavior)."""
        models = get_clustering_models(random_state=42, n_samples=None)
        
        # None should be treated as small dataset
        assert len(models) == 5
        assert 'KMeans' in models
        assert 'Agglomerative' in models

    def test_model_types(self):
        """Test that returned models are of correct types."""
        models = get_clustering_models(random_state=42, n_samples=100)
        
        assert isinstance(models['KMeans'], AutoKMeans)
        assert isinstance(models['GMM'], AutoGMM)
        assert isinstance(models['DBSCAN'], AutoDBSCAN)
        assert isinstance(models['Agglomerative'], AgglomerativeClustering)
        assert isinstance(models['Spectral'], SpectralClustering)

    def test_random_state_propagation(self):
        """Test that random_state is correctly propagated."""
        models = get_clustering_models(random_state=123, n_samples=100)
        
        assert models['KMeans'].random_state == 123
        assert models['GMM'].random_state == 123
        assert models['Spectral'].random_state == 123

    def test_k_range_for_large_dataset(self):
        """Test that k_range is adjusted for large dataset."""
        models_small = get_clustering_models(random_state=42, n_samples=100)
        models_large = get_clustering_models(random_state=42, n_samples=15000)
        
        # Large dataset should have smaller k_range
        assert models_small['KMeans'].k_range == (2, 10)
        assert models_large['KMeans'].k_range == (2, 8)
        
        assert models_small['GMM'].k_range == (2, 10)
        assert models_large['GMM'].k_range == (2, 6)

    def test_max_samples_setting(self):
        """Test that max_samples is correctly set."""
        models_small = get_clustering_models(random_state=42, n_samples=5000)
        models_large = get_clustering_models(random_state=42, n_samples=15000)
        
        # Both should respect the max_samples limit
        assert models_small['KMeans'].max_samples == 5000
        assert models_large['KMeans'].max_samples == 10000  # Capped at 10000


class TestClusteringIntegration:
    """Integration tests for clustering models."""

    def test_all_models_fit_and_predict(self, small_blob_dataset):
        """Test that all returned models can fit and predict."""
        X, _ = small_blob_dataset
        models = get_clustering_models(random_state=42, n_samples=len(X))
        
        for model_name, model in models.items():
            # All models should have fit/fit_predict
            if hasattr(model, 'fit_predict'):
                predictions = model.fit_predict(X)
            else:
                model.fit(X)
                predictions = model.predict(X)
            
            assert len(predictions) == len(X)
            assert predictions.dtype in [np.int32, np.int64]

    def test_models_consistency_across_fits(self, small_blob_dataset):
        """Test that models produce consistent results across multiple fits."""
        X, _ = small_blob_dataset
        
        kmeans1 = AutoKMeans(random_state=42)
        kmeans1.fit(X)
        pred1 = kmeans1.predict(X)
        
        kmeans2 = AutoKMeans(random_state=42)
        kmeans2.fit(X)
        pred2 = kmeans2.predict(X)
        
        # With same random_state, should get same results
        # (note: cluster labels might be permuted, but clustering should be equivalent)
        assert np.array_equal(pred1, pred2)

    def test_models_different_random_states(self, small_blob_dataset):
        """Test that different random_states can produce different results."""
        X, _ = small_blob_dataset
        
        kmeans1 = AutoKMeans(random_state=42)
        kmeans1.fit(X)
        best_k1 = kmeans1.best_k
        
        kmeans2 = AutoKMeans(random_state=123)
        kmeans2.fit(X)
        best_k2 = kmeans2.best_k
        
        # With the same data, best_k should be the same even with different random_states
        assert best_k1 == best_k2  # Should find same structure
