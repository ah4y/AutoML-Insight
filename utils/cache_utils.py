"""
Caching utilities for AutoML-Insight.

Provides decorators and utilities for caching expensive operations:
- Data loading and profiling
- Model training results
- Visualization data
- Feature engineering results
"""

import streamlit as st
import pandas as pd
import hashlib
import pickle
from typing import Any, Callable, Optional
from functools import wraps
import logging

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
def cached_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV with caching.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading CSV (will be cached): {file_path}")
    return pd.read_csv(file_path, **kwargs)


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
    
    if task_type.lower() == 'classification':
        from core.models_supervised import get_supervised_models
        return get_supervised_models(random_seed=random_seed)
    elif task_type.lower() == 'clustering':
        from core.models_clustering import get_clustering_models
        return get_clustering_models(random_seed=random_seed)
    else:
        raise ValueError(f"Invalid task type: {task_type}")


@st.cache_data(ttl=7200, show_spinner=False)
def cached_model_evaluation(
    model_name: str,
    data_hash: str,
    config_hash: str,
    _evaluator: Any,
    _model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Cache model evaluation results.
    
    Args:
        model_name: Name of the model
        data_hash: Hash of the data
        config_hash: Hash of configuration
        _evaluator: Evaluator instance (prefixed with _)
        _model: Model instance (prefixed with _)
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating {model_name} (will be cached)")
    return _evaluator.evaluate_model(_model, X, y, model_name)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_visualization_data(
    plot_type: str,
    data_hash: str,
    **plot_params
) -> dict:
    """
    Cache visualization data preparation.
    
    Args:
        plot_type: Type of plot
        data_hash: Hash of the data
        **plot_params: Parameters for plot generation
        
    Returns:
        Dictionary with plot data
    """
    logger.info(f"Preparing {plot_type} visualization data (will be cached)")
    # This is a placeholder - actual visualization preparation happens in calling code
    # This function ensures the cache key is properly constructed
    return {'cached': True, 'plot_type': plot_type, 'params': plot_params}


@st.cache_data(ttl=1800, show_spinner=False)
def cached_feature_importance(_explainer: Any, model_name: str, data_hash: str) -> dict:
    """
    Cache SHAP/feature importance calculations.
    
    Args:
        _explainer: ModelExplainer instance (prefixed with _)
        model_name: Name of the model
        data_hash: Hash of the data
        
    Returns:
        Dictionary with explanation results
    """
    logger.info(f"Computing feature importance for {model_name} (will be cached)")
    # Actual computation happens in calling code
    return {'model': model_name, 'cached': True}


def clear_cache(cache_type: str = 'all'):
    """
    Clear Streamlit caches.
    
    Args:
        cache_type: Type of cache to clear ('data', 'resource', 'all')
    """
    try:
        if cache_type in ('data', 'all'):
            st.cache_data.clear()
            logger.info("Cleared data cache")
        
        if cache_type in ('resource', 'all'):
            st.cache_resource.clear()
            logger.info("Cleared resource cache")
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")


def get_cache_stats() -> dict:
    """
    Get cache statistics (if available).
    
    Returns:
        Dictionary with cache stats
    """
    # Note: Streamlit doesn't expose direct cache stats
    # This is a placeholder for future implementation
    return {
        'cache_enabled': True,
        'data_cache': 'active',
        'resource_cache': 'active'
    }


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
        
        if dataset_name == 'iris':
            iris = load_iris()
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            data['target'] = iris.target
            return data, 'target'
        elif dataset_name == 'wine':
            wine = load_wine()
            data = pd.DataFrame(wine.data, columns=wine.feature_names)
            data['target'] = wine.target
            return data, 'target'
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


# Decorator for custom caching logic
def cache_expensive_operation(ttl: int = 3600):
    """
    Decorator for caching expensive operations.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @st.cache_data(ttl=ttl, show_spinner=f"Computing {func.__name__}...")
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage functions
def cache_ml_pipeline_stage(stage_name: str, data_hash: str):
    """
    Decorator factory for caching ML pipeline stages.
    
    Args:
        stage_name: Name of the pipeline stage
        data_hash: Hash of the input data
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @st.cache_data(ttl=3600, show_spinner=f"Processing {stage_name}...")
        def wrapper(*args, **kwargs):
            logger.info(f"Executing cached pipeline stage: {stage_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
