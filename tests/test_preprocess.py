"""Tests for preprocessing."""

import pytest
import numpy as np
import pandas as pd
from core.preprocess import DataPreprocessor


def test_preprocessor_basic(sample_data):
    """Test basic preprocessing."""
    X, y = sample_data
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Check shapes
    assert X_transformed.shape[0] == X.shape[0]
    assert y_transformed.shape[0] == y.shape[0]
    
    # Check no missing values
    assert not np.isnan(X_transformed).any()


def test_preprocessor_with_missing(sample_data_with_missing):
    """Test preprocessing with missing values."""
    X, y = sample_data_with_missing
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Missing values should be imputed
    assert not np.isnan(X_transformed).any()


def test_preprocessor_categorical():
    """Test preprocessing with categorical features."""
    X = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [5, 4, 3, 2, 1],
        'category': ['A', 'B', 'A', 'B', 'C']
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # One-hot encoding should increase dimensions
    assert X_transformed.shape[1] > X.shape[1]


def test_preprocessor_transform(sample_data):
    """Test transform on new data."""
    X, y = sample_data
    
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(X, y)
    
    # Transform new data
    X_new = X.iloc[:10]
    X_new_transformed = preprocessor.transform(X_new)
    
    assert X_new_transformed.shape[1] == preprocessor.get_feature_names().__len__()
