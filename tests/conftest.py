"""Unit tests for AutoML-Insight."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


@pytest.fixture
def sample_data():
    """Fixture providing sample dataset."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y


@pytest.fixture
def sample_data_with_missing():
    """Fixture providing dataset with missing values."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Introduce missing values
    X.iloc[0:5, 0] = np.nan
    X.iloc[10:15, 1] = np.nan
    
    return X, y
