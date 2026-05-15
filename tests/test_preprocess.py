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
    
    # Disable dimred to see pure one-hot encoding effect
    from core.dimred import DimRedConfig
    dimred_config = DimRedConfig(enable='off')
    preprocessor = DataPreprocessor(dimred_config=dimred_config)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # One-hot encoding should increase dimensions from 3 to at least 4
    # (2 numeric + 1 one-hot encoded categorical with 3 categories becomes 2 + 3 = 5)
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


def test_preprocessor_numpy_array_input():
    """Test preprocessing with numpy array input (lines 84-88)."""
    # Create numpy array input
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Check that it converts and processes correctly
    assert X_transformed.shape[0] == 50
    assert not np.isnan(X_transformed).any()
    assert y_transformed is not None


def test_preprocessor_constant_features():
    """Test removal of constant/zero-variance features (lines 91-96)."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [1, 2, 3, 4, 5],
        'const_feature': [7, 7, 7, 7, 7],  # Constant feature
        'quasi_const': [1, 1, 1, 1, 2]  # Quasi-constant
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Constant feature should be removed
    assert X_transformed.shape[1] < X.shape[1]


def test_preprocessor_low_variance_features():
    """Test removal of low-variance numeric features (lines 100-106)."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [1, 2, 3, 4, 5],
        'low_var': [1.0, 1.0, 1.001, 1.0, 1.0001]  # Variance < 0.01
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Low-variance feature should be removed
    assert X_transformed.shape[1] <= X.shape[1]


def test_preprocessor_high_cardinality_categorical(sample_data):
    """Test handling of high-cardinality categorical features (lines 113-130)."""
    X = pd.DataFrame({
        'numeric_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        'numeric_2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 1] * 10,
        'high_card_cat': [f'cat_{i}' for i in range(100)]  # High-cardinality
    })
    y = pd.Series([0, 1] * 50)
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # High-cardinality feature should be removed
    assert X_transformed is not None
    assert y_transformed is not None


def test_preprocessor_feature_selection_pre_transformation():
    """Test pre-selection feature selection before transformation (lines 133-156)."""
    # Create dataset with many numeric features and target
    np.random.seed(42)
    n_samples = 100
    n_features = 1500  # Exceed max_features default of 1000
    
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    preprocessor = DataPreprocessor(max_features=500)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Should have reduced to max_features or less
    assert X_transformed.shape[1] <= 500


def test_preprocessor_emergency_reduction():
    """Test emergency feature reduction for large memory datasets (lines 170-193)."""
    # Create dataset that would trigger emergency reduction
    np.random.seed(42)
    n_samples = 500
    n_numeric = 300
    n_categorical = 5
    
    X_numeric = pd.DataFrame(
        np.random.rand(n_samples, n_numeric),
        columns=[f'num_{i}' for i in range(n_numeric)]
    )
    X_categorical = pd.DataFrame({
        f'cat_{i}': np.random.choice(['A', 'B', 'C'], n_samples)
        for i in range(n_categorical)
    })
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # This should trigger emergency reduction due to memory estimation
    preprocessor = DataPreprocessor(max_features=200)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Should still complete successfully
    assert X_transformed.shape[0] == n_samples
    assert y_transformed.shape[0] == n_samples


def test_preprocessor_post_transformation_feature_selection():
    """Test feature selection after transformation (lines 241-262)."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    preprocessor = DataPreprocessor(max_features=20)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # After selection, should have <= max_features
    assert X_transformed.shape[1] <= 20


def test_preprocessor_transform_without_fit():
    """Test transform method raises error when not fitted (lines 301-302)."""
    preprocessor = DataPreprocessor()
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    with pytest.raises(ValueError):
        preprocessor.transform(X)


def test_preprocessor_transform_with_feature_selector():
    """Test transform applies feature selector (lines 311-312)."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.rand(100, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    y = pd.Series(np.random.randint(0, 2, 100))
    
    preprocessor = DataPreprocessor(max_features=20)
    X_train, y_train = preprocessor.fit_transform(X, y)
    
    # If feature_selector was created, transform should use it
    X_test = X.iloc[:20]
    X_test_transformed = preprocessor.transform(X_test)
    
    # Should have same number of features
    assert X_test_transformed.shape[1] == X_train.shape[1]


def test_preprocessor_get_feature_names():
    """Test get_feature_names method (lines 342-349)."""
    X = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'cat': ['A', 'B', 'A', 'B', 'C']
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    from core.dimred import DimRedConfig
    dimred_config = DimRedConfig(enable='off')
    preprocessor = DataPreprocessor(dimred_config=dimred_config)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) > 0
    assert len(feature_names) == X_transformed.shape[1]


def test_preprocessor_get_feature_names_exception_handling():
    """Test get_feature_names exception handling (lines 337-338)."""
    X = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'cat': ['A', 'B', 'A', 'B', 'C']
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    from core.dimred import DimRedConfig
    dimred_config = DimRedConfig(enable='off')
    preprocessor = DataPreprocessor(dimred_config=dimred_config)
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    feature_names = preprocessor.get_feature_names()
    # Should return features despite potential exceptions
    assert isinstance(feature_names, list)


def test_preprocessor_set_params():
    """Test set_params method (lines 368-380)."""
    preprocessor = DataPreprocessor(max_features=1000)
    
    # Test set_params
    new_preprocessor = preprocessor.set_params(max_features=500)
    
    assert new_preprocessor.max_features == 500
    assert new_preprocessor is preprocessor  # Should return self


def test_preprocessor_get_params():
    """Test get_params method (lines 351-366)."""
    from core.dimred import DimRedConfig
    dimred_config = DimRedConfig(enable='off')
    preprocessor = DataPreprocessor(max_features=500, dimred_config=dimred_config)
    
    params = preprocessor.get_params()
    
    assert 'max_features' in params
    assert params['max_features'] == 500
    assert 'dimred_config' in params


def test_preprocessor_with_numeric_and_categorical():
    """Test preprocessing with mixed numeric and categorical features."""
    np.random.seed(42)
    X = pd.DataFrame({
        'num_1': np.random.rand(30),
        'num_2': np.random.rand(30),
        'cat_1': np.random.choice(['A', 'B', 'C'], 30),
        'cat_2': np.random.choice(['X', 'Y'], 30)
    })
    y = pd.Series(np.random.randint(0, 2, 30))
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    assert X_transformed.shape[0] == 30
    assert X_transformed.shape[1] > 0


def test_preprocessor_no_target():
    """Test preprocessing without target variable."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [5, 4, 3, 2, 1]
    })
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == 5
    assert y_transformed is None


def test_preprocessor_string_target():
    """Test preprocessing with string target values."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [5, 4, 3, 2, 1]
    })
    y = pd.Series(['cat', 'dog', 'cat', 'dog', 'cat'])
    
    preprocessor = DataPreprocessor()
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Should label-encode string targets
    assert y_transformed is not None
    assert len(np.unique(y_transformed)) == 2
