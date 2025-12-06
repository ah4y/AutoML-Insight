"""Unit tests for core/dimred.py module.

Tests the dimensionality reduction factory, configuration, and auto-selection logic.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_sparse_coded_signal
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix

from core.dimred import (
    DimRedConfig,
    make_dimred,
    should_enable_dimred,
    select_dimred_method,
    DimRedSelector
)


class TestDimRedConfig:
    """Test DimRedConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DimRedConfig()
        
        assert config.enable == 'auto'
        assert config.method == 'auto'
        assert config.variance_target == 0.95
        assert config.k_max == 256
        assert config.whiten is True
        assert config.seed == 42
    
    def test_custom_config(self):
        """Test custom configuration."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'variance_target': 0.90,
                'k_max': 128,
                'whiten': False,
                'seed': 123
            }
        }
        
        config = DimRedConfig(config_dict)
        
        assert config.enable == 'on'
        assert config.method == 'pca'
        assert config.variance_target == 0.90
        assert config.k_max == 128
        assert config.whiten is False
        assert config.seed == 123
    
    def test_invalid_enable(self):
        """Test invalid enable value."""
        config_dict = {
            'dimred': {
                'enable': 'invalid'
            }
        }
        
        with pytest.raises(ValueError, match="Invalid enable value"):
            DimRedConfig(config_dict)
    
    def test_invalid_method(self):
        """Test invalid method value."""
        config_dict = {
            'dimred': {
                'method': 'invalid'
            }
        }
        
        with pytest.raises(ValueError, match="Invalid method value"):
            DimRedConfig(config_dict)
    
    def test_invalid_variance_target(self):
        """Test invalid variance_target value."""
        config_dict = {
            'dimred': {
                'variance_target': 1.5  # Too high
            }
        }
        
        with pytest.raises(ValueError, match="variance_target must be in"):
            DimRedConfig(config_dict)
    
    def test_invalid_k_max(self):
        """Test invalid k_max value."""
        config_dict = {
            'dimred': {
                'k_max': 1  # Too low
            }
        }
        
        with pytest.raises(ValueError, match="k_max must be >= 2"):
            DimRedConfig(config_dict)


class TestShouldEnableDimred:
    """Test should_enable_dimred function."""
    
    def test_enable_on_mode(self):
        """Test enable='on' mode always returns True."""
        assert should_enable_dimred(10, 100, False, "on") is True
        assert should_enable_dimred(1000, 10, True, "on") is True
    
    def test_disable_off_mode(self):
        """Test enable='off' mode always returns False."""
        assert should_enable_dimred(10, 100, False, "off") is False
        assert should_enable_dimred(1000, 10, True, "off") is False
    
    def test_auto_mode_high_dimensional(self):
        """Test auto mode enables for high-dimensional data."""
        # Many features (>100)
        assert should_enable_dimred(1500, 100, False, "auto") is True
        
        # High feature-to-sample ratio
        assert should_enable_dimred(50, 100, False, "auto") is True
    
    def test_auto_mode_sparse_data(self):
        """Test auto mode enables for sparse data."""
        assert should_enable_dimred(500, 100, True, "auto") is True
    
    def test_auto_mode_low_dimensional(self):
        """Test auto mode disables for low-dimensional data."""
        # Few features, many samples
        assert should_enable_dimred(10, 1000, False, "auto") is False
        
        # Low feature-to-sample ratio, not sparse
        assert should_enable_dimred(20, 1000, False, "auto") is False


class TestSelectDimredMethod:
    """Test select_dimred_method function."""
    
    def test_sparse_data_tsvd(self):
        """Test TruncatedSVD selection for sparse data."""
        method = select_dimred_method(True, 100, "auto")
        assert method == "tsvd"
    
    def test_large_dense_data_ipca(self):
        """Test IncrementalPCA for large dense datasets."""
        method = select_dimred_method(False, 50000, "auto")  # Large n_features 
        assert method == "ipca"
    
    def test_moderate_dense_data_pca(self):
        """Test PCA for moderate dense datasets."""
        method = select_dimred_method(False, 100, "auto")  # Moderate size
        assert method == "pca"
    
    def test_small_dense_data_pca(self):
        """Test PCA for small dense datasets.""" 
        method = select_dimred_method(False, 50, "auto")
        assert method == "pca"


class TestMakeDimred:
    """Test make_dimred factory function."""
    
    def test_pca_creation(self):
        """Test PCA transformer creation."""
        config_dict = {
            'dimred': {
                'method': 'pca',
                'k_max': 10,
                'whiten': True,
                'seed': 42
            }
        }
        config = DimRedConfig(config_dict)
        
        transformer = make_dimred(
            is_sparse_after_ohe=False,
            n_features=20,
            n_samples=100,
            cfg=config
        )
        
        assert isinstance(transformer, PCA)
        assert transformer.whiten is True
        assert transformer.random_state == 42
    
    def test_tsvd_creation(self):
        """Test TruncatedSVD transformer creation."""
        transformer = make_dimred(
            method="tsvd",
            n_components=20,
            random_state=42
        )
        
        assert isinstance(transformer, TruncatedSVD)
        assert transformer.n_components == 20
        assert transformer.random_state == 42
    
    def test_ipca_creation(self):
        """Test IncrementalPCA transformer creation."""
        transformer = make_dimred(
            method="ipca", 
            n_components=15,
            whiten=False
        )
        
        assert isinstance(transformer, IncrementalPCA)
        assert transformer.n_components == 15
        assert transformer.whiten is False
    
    def test_invalid_method(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dimred method"):
            make_dimred(method="invalid")
    
    def test_pca_with_variance_target(self):
        """Test PCA creation with variance_target instead of n_components."""
        # Create sample data
        X = np.random.randn(100, 20)
        
        transformer = make_dimred(
            method="pca",
            variance_target=0.95,
            whiten=False,
            random_state=42
        )
        
        assert isinstance(transformer, PCA)
        assert transformer.n_components == 0.95  # Should set variance target
    
    def test_auto_method_selection(self):
        """Test auto method selection."""
        # Dense data -> should get PCA
        X_dense = np.random.randn(100, 20)
        transformer = make_dimred(
            method="auto",
            n_components=10,
            data_info={'is_sparse': False, 'n_samples': 100}
        )
        assert isinstance(transformer, PCA)
        
        # Sparse indication -> should get TruncatedSVD
        transformer_sparse = make_dimred(
            method="auto", 
            n_components=10,
            data_info={'is_sparse': True, 'n_samples': 100}
        )
        assert isinstance(transformer_sparse, TruncatedSVD)


class TestDimRedSelector:
    """Test DimRedSelector runtime wrapper."""
    
    def test_initialization(self):
        """Test DimRedSelector initialization."""
        config = DimRedConfig()
        selector = DimRedSelector(config)
        
        assert selector.config == config
        assert selector.is_fitted is False
        assert selector.transformer_ is None
        assert selector.selected_method_ is None
    
    def test_fit_enabled(self):
        """Test fitting when dimred is enabled."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'variance_target': 0.95
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        # Create sample data
        X = np.random.randn(100, 20)
        
        selector.fit(X)
        
        assert selector.is_fitted is True
        assert isinstance(selector.transformer_, PCA)
        assert selector.selected_method_ == "pca"
    
    def test_fit_disabled(self):
        """Test fitting when dimred is disabled."""
        config_dict = {
            'dimred': {
                'enable': 'off'
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X = np.random.randn(100, 20)
        selector.fit(X)
        
        assert selector.is_fitted is True
        assert selector.transformer_ is None
        assert selector.selected_method_ is None
    
    def test_transform_enabled(self):
        """Test transform when dimred is enabled."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'k_max': 10  # Limit components
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        # Create sample data
        X = np.random.randn(100, 20)
        
        selector.fit(X)
        X_transformed = selector.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]  # Same number of samples
        assert X_transformed.shape[1] <= 10  # Reduced dimensions
        assert X_transformed.shape[1] < X.shape[1]  # Actually reduced
    
    def test_transform_disabled(self):
        """Test transform when dimred is disabled."""
        config_dict = {
            'dimred': {
                'enable': 'off'
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X = np.random.randn(100, 20)
        
        selector.fit(X)
        X_transformed = selector.transform(X)
        
        # Should return original data unchanged
        np.testing.assert_array_equal(X_transformed, X)
    
    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'k_max': 5
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X = np.random.randn(100, 20)
        X_transformed = selector.fit_transform(X)
        
        assert selector.is_fitted is True
        assert X_transformed.shape[1] <= 5
    
    def test_auto_mode_selection(self):
        """Test auto mode selects appropriate method."""
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto'
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        # High-dimensional data should trigger dimred
        X = np.random.randn(100, 200)  # Many features
        selector.fit(X)
        
        # Should have selected a method
        if selector.transformer_ is not None:
            assert selector.selected_method_ in ["pca", "tsvd", "ipca"]
    
    def test_sparse_data_handling(self):
        """Test handling of sparse data."""
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'auto'
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        # Create sparse matrix
        X_dense = np.random.randn(100, 50)
        X_dense[X_dense < 1] = 0  # Make sparse
        X_sparse = csr_matrix(X_dense)
        
        selector.fit(X_sparse)
        
        # Should select TruncatedSVD for sparse data
        if selector.transformer_ is not None:
            assert isinstance(selector.transformer_, TruncatedSVD)
            assert selector.selected_method_ == "tsvd"
    
    def test_pipeline_integration(self):
        """Test integration with sklearn Pipeline."""
        config_dict = {
            'dimred': {
                'enable': 'on', 
                'method': 'pca',
                'k_max': 10
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        # Create pipeline
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
            ('dimred', selector),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Create sample classification data
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        
        # Fit and predict
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
        assert selector.is_fitted is True


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_high_dimensional_dense_data(self):
        """Test with high-dimensional dense dataset."""
        # Create high-dimensional data
        X, y = make_classification(
            n_samples=200,
            n_features=500,
            n_informative=50,
            n_redundant=50,
            random_state=42
        )
        
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto',
                'variance_target': 0.95
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X_transformed = selector.fit_transform(X)
        
        # Should reduce dimensions significantly
        assert X_transformed.shape[1] < X.shape[1]
        assert X_transformed.shape[0] == X.shape[0]  # Same samples
        assert selector.is_fitted is True
    
    def test_small_dataset(self):
        """Test with small dataset where dimred might be skipped."""
        # Small, low-dimensional data
        X = np.random.randn(50, 5)
        
        config_dict = {
            'dimred': {
                'enable': 'auto',
                'method': 'auto'
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X_transformed = selector.fit_transform(X)
        
        # Should probably skip dimred for small, low-dim data
        # (depending on the auto logic)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_forced_pca_on_appropriate_data(self):
        """Test forcing PCA on data where it makes sense."""
        X = np.random.randn(100, 50)
        
        config_dict = {
            'dimred': {
                'enable': 'on',
                'method': 'pca',
                'k_max': 20,
                'whiten': True
            }
        }
        config = DimRedConfig(config_dict)
        selector = DimRedSelector(config)
        
        X_transformed = selector.fit_transform(X)
        
        assert isinstance(selector.transformer_, PCA)
        assert X_transformed.shape[1] <= 20
        assert selector.transformer_.whiten is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])