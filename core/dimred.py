"""Dimensionality reduction factory and utilities for AutoML-Insight.

This module provides leakage-safe dimensionality reduction components that can be
integrated into scikit-learn pipelines for preprocessing. Supports PCA, TruncatedSVD,
and IncrementalPCA with intelligent auto-selection based on data characteristics.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from utils.logging_utils import setup_logger


class DimRedConfig:
    """Configuration class for dimensionality reduction parameters."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with configuration dictionary or defaults."""
        if config_dict is None:
            config_dict = {}
        
        # Extract dimred config with safe defaults
        dimred_config = config_dict.get('dimred', {})
        
        self.enable = dimred_config.get('enable', 'auto')
        self.method = dimred_config.get('method', 'auto')
        self.variance_target = dimred_config.get('variance_target', 0.95)
        self.k_max = dimred_config.get('k_max', 256)
        self.whiten = dimred_config.get('whiten', True)
        self.seed = dimred_config.get('seed', 42)
        
        # Validation
        if self.enable not in ['off', 'on', 'auto']:
            raise ValueError(f"Invalid enable value: {self.enable}")
        if self.method not in ['pca', 'tsvd', 'ipca', 'auto']:
            raise ValueError(f"Invalid method value: {self.method}")
        if not 0.1 <= self.variance_target <= 0.99:
            raise ValueError(f"variance_target must be in [0.1, 0.99], got {self.variance_target}")
        if self.k_max < 2:
            raise ValueError(f"k_max must be >= 2, got {self.k_max}")


def should_enable_dimred(
    n_features: int,
    n_samples: int,
    is_sparse_after_ohe: bool,
    enable_mode: str
) -> bool:
    """
    Determine if dimensionality reduction should be enabled based on data characteristics.
    
    Args:
        n_features: Number of features in the dataset
        n_samples: Number of samples in the dataset
        is_sparse_after_ohe: Whether data will be sparse after one-hot encoding
        enable_mode: Configuration mode ('off', 'on', 'auto')
    
    Returns:
        Boolean indicating whether to enable dimensionality reduction
    """
    if enable_mode == 'off':
        return False
    elif enable_mode == 'on':
        return True
    elif enable_mode == 'auto':
        # Auto-enable heuristics
        feature_to_sample_ratio = n_features / max(n_samples, 1)
        
        # Enable if:
        # 1. High-dimensional data (many features relative to samples)
        # 2. Very high feature count
        # 3. Sparse data that could benefit from dimensionality reduction
        return (
            feature_to_sample_ratio > 0.5 or  # More than half as many features as samples
            n_features > 1000 or              # High absolute feature count
            (is_sparse_after_ohe and n_features > 100)  # Sparse data with moderate feature count
        )
    else:
        raise ValueError(f"Invalid enable_mode: {enable_mode}")


def select_dimred_method(
    is_sparse_after_ohe: bool,
    n_features: int,
    method_mode: str
) -> str:
    """
    Select the appropriate dimensionality reduction method based on data characteristics.
    
    Args:
        is_sparse_after_ohe: Whether data will be sparse after one-hot encoding
        n_features: Number of features in the dataset
        method_mode: Configuration mode ('pca', 'tsvd', 'ipca', 'auto')
    
    Returns:
        Selected method name ('pca', 'tsvd', or 'ipca')
    """
    if method_mode in ['pca', 'tsvd', 'ipca']:
        return method_mode
    elif method_mode == 'auto':
        # Auto-selection logic
        if is_sparse_after_ohe:
            # TruncatedSVD works well with sparse matrices
            return 'tsvd'
        elif n_features > 50000:
            # IncrementalPCA for very high-dimensional dense data
            return 'ipca'
        else:
            # Standard PCA for moderate-size dense data
            return 'pca'
    else:
        raise ValueError(f"Invalid method_mode: {method_mode}")


def make_dimred(
    is_sparse_after_ohe: bool,
    n_features: int,
    n_samples: int,
    cfg: DimRedConfig
) -> Optional[BaseEstimator]:
    """
    Factory function to create dimensionality reduction transformer.
    
    Args:
        is_sparse_after_ohe: Whether the data will be sparse after preprocessing
        n_features: Number of features in the dataset
        n_samples: Number of samples in the dataset  
        cfg: DimRedConfig instance with parameters
    
    Returns:
        Configured transformer instance or None if disabled
    """
    logger = setup_logger()
    
    # Check if dimensionality reduction should be enabled
    if not should_enable_dimred(n_features, n_samples, is_sparse_after_ohe, cfg.enable):
        logger.info("Dimensionality reduction disabled")
        return None
    
    # Select the appropriate method
    method = select_dimred_method(is_sparse_after_ohe, n_features, cfg.method)
    
    logger.info(f"Selected dimensionality reduction method: {method}")
    
    # Create the transformer based on selected method
    if method == 'pca':
        # Use randomized SVD for efficiency with dense matrices
        transformer = PCA(
            n_components=cfg.variance_target,  # Will keep components explaining this variance
            svd_solver='randomized',
            whiten=cfg.whiten,
            random_state=cfg.seed
        )
        logger.info(f"Created PCA with {cfg.variance_target:.1%} variance target")
        
    elif method == 'tsvd':
        # TruncatedSVD for sparse matrices
        # Calculate reasonable number of components
        n_components = min(
            cfg.k_max,
            max(2, int(np.sqrt(n_features))),  # Square root heuristic
            n_features - 1,  # Can't exceed n_features - 1
            n_samples - 1    # Can't exceed n_samples - 1
        )
        
        transformer = TruncatedSVD(
            n_components=n_components,
            random_state=cfg.seed
        )
        logger.info(f"Created TruncatedSVD with {n_components} components")
        
    elif method == 'ipca':
        # IncrementalPCA for very large datasets
        n_components = min(
            cfg.k_max,
            max(2, int(np.sqrt(n_features))),
            n_features - 1,
            n_samples - 1
        )
        
        transformer = IncrementalPCA(
            n_components=n_components
        )
        logger.info(f"Created IncrementalPCA with {n_components} components")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return transformer


class DimRedSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper that determines whether to apply dimensionality reduction at runtime.
    
    This is useful when the decision to apply dimred depends on the actual
    preprocessed data characteristics that are only known after fitting.
    """
    
    def __init__(self, cfg: DimRedConfig):
        """Initialize with configuration."""
        self.cfg = cfg
        self.dimred_transformer = None
        self.logger = setup_logger()
    
    def fit(self, X, y=None):
        """
        Fit the dimensionality reduction transformer based on data characteristics.
        
        Args:
            X: Input data (after preprocessing)
            y: Target variable (unused)
        
        Returns:
            self
        """
        from scipy import sparse
        
        # Determine data characteristics
        n_samples, n_features = X.shape
        is_sparse = sparse.issparse(X)
        
        # Create appropriate transformer
        self.dimred_transformer = make_dimred(
            is_sparse_after_ohe=is_sparse,
            n_features=n_features,
            n_samples=n_samples,
            cfg=self.cfg
        )
        
        # Fit the transformer if created
        if self.dimred_transformer is not None:
            self.logger.info(f"Fitting {type(self.dimred_transformer).__name__} on {n_samples}x{n_features} data")
            self.dimred_transformer.fit(X, y)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using fitted dimensionality reduction.
        
        Args:
            X: Input data
        
        Returns:
            Transformed data or original data if no reduction applied
        """
        if self.dimred_transformer is not None:
            X_transformed = self.dimred_transformer.transform(X)
            self.logger.info(f"Dimensionality reduced from {X.shape[1]} to {X_transformed.shape[1]} features")
            return X_transformed
        else:
            return X
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformed data."""
        if self.dimred_transformer is not None:
            if hasattr(self.dimred_transformer, 'get_feature_names_out'):
                return self.dimred_transformer.get_feature_names_out(input_features)
            else:
                # Generate generic component names
                n_components = self.dimred_transformer.n_components_
                method_name = type(self.dimred_transformer).__name__.lower()
                return [f"{method_name}_component_{i}" for i in range(n_components)]
        else:
            return input_features
    
    def get_n_components(self) -> int:
        """Get the number of components after dimensionality reduction."""
        if self.dimred_transformer is not None:
            return getattr(self.dimred_transformer, 'n_components_', 0)
        else:
            return 0
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratios if available."""
        if (self.dimred_transformer is not None and 
            hasattr(self.dimred_transformer, 'explained_variance_ratio_')):
            return self.dimred_transformer.explained_variance_ratio_
        return None


def load_dimred_config(config_path: Optional[str] = None) -> DimRedConfig:
    """
    Load dimensionality reduction configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        DimRedConfig instance
    """
    if config_path is None:
        return DimRedConfig()
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return DimRedConfig(config_dict)
    except Exception as e:
        logger = setup_logger()
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return DimRedConfig()