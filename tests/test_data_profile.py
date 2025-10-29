"""Tests for data profiling."""

import pytest
import numpy as np
import pandas as pd
from core.data_profile import DataProfiler


def test_data_profiler_basic(sample_data):
    """Test basic data profiling."""
    X, y = sample_data
    
    profiler = DataProfiler()
    profile = profiler.profile_dataset(X, y)
    
    # Check basic metrics
    assert profile['n_samples'] == 150
    assert profile['n_features'] == 4
    assert profile['task_type'] == 'classification'
    assert profile['n_classes'] == 3


def test_data_profiler_with_missing(sample_data_with_missing):
    """Test profiling with missing values."""
    X, y = sample_data_with_missing
    
    profiler = DataProfiler()
    profile = profiler.profile_dataset(X, y)
    
    # Check missing value detection
    assert profile['missing_ratio'] > 0
    assert profile['features_with_missing'] > 0


def test_data_profiler_clustering():
    """Test profiling for clustering task."""
    X = pd.DataFrame(np.random.randn(100, 5))
    
    profiler = DataProfiler()
    profile = profiler.profile_dataset(X)
    
    assert profile['task_type'] == 'clustering'
    assert profile['n_samples'] == 100
    assert profile['n_features'] == 5


def test_profile_vector_generation(sample_data):
    """Test meta-feature vector generation."""
    X, y = sample_data
    
    profiler = DataProfiler()
    profiler.profile_dataset(X, y)
    
    vector = profiler.get_profile_vector()
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
