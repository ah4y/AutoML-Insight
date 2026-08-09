"""
Caching utilities for AutoML-Insight.

Provides decorators and utilities for caching expensive operations:
- Data loading and profiling
- Model training results
- Visualization data
- Feature engineering results
"""

import hashlib
import logging
from typing import Any, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate a hash for a DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        Hash string
    """
    try:
        # Use shape and first/last rows for hash
        hash_input = f"{df.shape}_{df.head(5).to_json()}_{df.tail(5).to_json()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash DataFrame: {e}")
        # Fallback to simple hash
        return hashlib.md5(str(df.shape).encode()).hexdigest()


def hash_params(**params) -> str:
    """
    Generate hash from parameters.

    Args:
        **params: Parameters to hash

    Returns:
        Hash string
    """
    try:
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash params: {e}")
        return hashlib.md5(str(params).encode()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_data_profile(_profiler: Any, data_hash: str, X: pd.DataFrame, y: Optional[pd.Series] = None) -> dict:
    """
    Cache data profiling results.

    Args:
        _profiler: DataProfiler instance (prefixed with _ to prevent hashing)
        data_hash: Hash of the data
        X: Feature DataFrame
        y: Target Series (optional)

    Returns:
        Profile dictionary
    """
    logger.info("Profiling data (will be cached)")
    return _profiler.profile_dataset(X, y)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_preprocess(_preprocessor: Any, data_hash: str, X: pd.DataFrame, y: Optional[pd.Series] = None) -> tuple:
    """
    Cache preprocessing results.

    Args:
        _preprocessor: DataPreprocessor instance (prefixed with _ to prevent hashing)
        data_hash: Hash of the data
        X: Feature DataFrame
        y: Target Series (optional)

    Returns:
        Tuple of (X_processed, y_processed)
    """
    logger.info("Preprocessing data (will be cached)")
    return _preprocessor.fit_transform(X, y)


@st.cache_resource(show_spinner=False)
def cached_get_models(task_type: str, random_seed: int = 42) -> dict:
    """
    Cache model initialization.

    Args:
        task_type: 'classification' or 'clustering'
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary of initialized models
    """
    logger.info(f"Initializing {task_type} models (will be cached)")

    if task_type.lower() == "classification":
        from core.models_supervised import get_supervised_models

        return get_supervised_models(random_seed=random_seed)
    elif task_type.lower() == "clustering":
        from core.models_clustering import get_clustering_models

        return get_clustering_models(random_seed=random_seed)
    else:
        raise ValueError(f"Invalid task type: {task_type}")


def clear_cache(cache_type: str = "all"):
    """
    Clear Streamlit caches.

    Args:
        cache_type: Type of cache to clear ('data', 'resource', 'all')
    """
    try:
        if cache_type in ("data", "all"):
            st.cache_data.clear()
            logger.info("Cleared data cache")

        if cache_type in ("resource", "all"):
            st.cache_resource.clear()
            logger.info("Cleared resource cache")

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")


class CachedDataLoader:
    """Wrapper for data loading with automatic caching."""

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Loading data...")
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV with caching."""
        return pd.read_csv(file_path, **kwargs)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Loading demo data...")
    def load_demo_dataset(dataset_name: str) -> tuple:
        """
        Load demo dataset with caching.

        Args:
            dataset_name: 'iris' or 'wine'

        Returns:
            Tuple of (data, target_column)
        """
        from sklearn.datasets import load_iris, load_wine

        if dataset_name == "iris":
            iris = load_iris()
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            data["target"] = iris.target
            return data, "target"
        elif dataset_name == "wine":
            wine = load_wine()
            data = pd.DataFrame(wine.data, columns=wine.feature_names)
            data["target"] = wine.target
            return data, "target"
        else:
            raise ValueError(f"Unknown demo dataset: {dataset_name}")


class CachedModelCache:
    """Manage trained model caching."""

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def cache_trained_model(model_name: str, data_hash: str, config_hash: str, _model: Any) -> Any:
        """
        Cache a trained model.

        Args:
            model_name: Name of the model
            data_hash: Hash of training data
            config_hash: Hash of configuration
            _model: Trained model instance (prefixed with _)

        Returns:
            The model (for chaining)
        """
        logger.info(f"Cached trained model: {model_name}")
        return _model

    @staticmethod
    def clear_model_cache():
        """Clear all cached models."""
        st.cache_resource.clear()
        logger.info("Cleared model cache")
