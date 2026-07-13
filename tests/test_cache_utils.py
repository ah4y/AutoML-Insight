"""
Tests for caching utilities.

Tests cache functionality for:
- Data loading
- Data profiling
- Preprocessing
- Model caching
- Visualization data
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache_utils import (
    CachedDataLoader,
    CachedModelCache,
    cached_read_csv,
    clear_cache,
    get_cache_stats,
    hash_dataframe,
    hash_params,
)


class TestHashFunctions:
    """Test hash generation functions."""

    def test_hash_dataframe_consistency(self):
        """Test that same DataFrame produces same hash."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})

        hash1 = hash_dataframe(df)
        hash2 = hash_dataframe(df)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_hash_dataframe_different_data(self):
        """Test that different DataFrames produce different hashes."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})  # Different last value

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        assert hash1 != hash2

    def test_hash_params_consistency(self):
        """Test parameter hashing consistency."""
        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

        hash1 = hash_params(**params)
        hash2 = hash_params(**params)

        assert hash1 == hash2
        assert isinstance(hash1, str)

    def test_hash_params_order_independence(self):
        """Test that parameter order doesn't affect hash."""
        hash1 = hash_params(a=1, b=2, c=3)
        hash2 = hash_params(c=3, a=1, b=2)

        assert hash1 == hash2

    def test_hash_params_different_values(self):
        """Test that different parameter values produce different hashes."""
        hash1 = hash_params(n_estimators=100)
        hash2 = hash_params(n_estimators=200)

        assert hash1 != hash2


class TestCachedDataLoader:
    """Test cached data loading."""

    def test_load_demo_iris(self):
        """Test loading Iris dataset."""
        data, target = CachedDataLoader.load_demo_dataset("iris")

        assert isinstance(data, pd.DataFrame)
        assert isinstance(target, str)
        assert target == "target"
        assert "target" in data.columns
        assert data.shape[0] == 150  # Iris has 150 samples
        assert data.shape[1] == 5  # 4 features + 1 target

    def test_load_demo_wine(self):
        """Test loading Wine dataset."""
        data, target = CachedDataLoader.load_demo_dataset("wine")

        assert isinstance(data, pd.DataFrame)
        assert isinstance(target, str)
        assert target == "target"
        assert "target" in data.columns
        assert data.shape[0] == 178  # Wine has 178 samples
        assert data.shape[1] == 14  # 13 features + 1 target

    def test_load_invalid_demo_dataset(self):
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError, match="Unknown demo dataset"):
            CachedDataLoader.load_demo_dataset("invalid_dataset")

    def test_load_csv_from_file(self):
        """Test loading CSV from file."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            temp_path = f.name

        try:
            data = CachedDataLoader.load_csv(temp_path)

            assert isinstance(data, pd.DataFrame)
            assert data.shape == (2, 3)
            assert list(data.columns) == ["col1", "col2", "col3"]
            assert data["col1"].tolist() == [1, 4]
        finally:
            os.unlink(temp_path)


class TestCacheUtilities:
    """Test cache management utilities."""

    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        stats = get_cache_stats()

        assert isinstance(stats, dict)
        assert "cache_enabled" in stats
        assert stats["cache_enabled"] is True

    def test_clear_cache_all(self):
        """Test clearing all caches."""
        # This should not raise an error
        clear_cache("all")

    def test_clear_cache_data(self):
        """Test clearing data cache."""
        clear_cache("data")

    def test_clear_cache_resource(self):
        """Test clearing resource cache."""
        clear_cache("resource")


class TestCachedModelCache:
    """Test model caching functionality."""

    def test_cache_trained_model(self):
        """Test caching a trained model."""

        # Create a mock model
        class MockModel:
            def __init__(self):
                self.trained = True

        model = MockModel()
        model_name = "test_model"
        data_hash = "abc123"
        config_hash = "def456"

        # Cache the model
        cached_model = CachedModelCache.cache_trained_model(model_name, data_hash, config_hash, model)

        assert cached_model is model
        assert cached_model.trained is True

    def test_clear_model_cache(self):
        """Test clearing model cache."""
        # This should not raise an error
        CachedModelCache.clear_model_cache()


class TestIntegration:
    """Integration tests for caching."""

    def test_end_to_end_data_loading_and_hashing(self):
        """Test complete workflow: load data -> hash -> cache."""
        # Load demo data
        data, target = CachedDataLoader.load_demo_dataset("iris")

        # Hash the data
        data_hash = hash_dataframe(data)

        # Verify
        assert isinstance(data_hash, str)
        assert len(data_hash) == 32

        # Load again - should come from cache (same hash)
        data2, target2 = CachedDataLoader.load_demo_dataset("iris")
        data_hash2 = hash_dataframe(data2)

        assert data_hash == data_hash2

    def test_parameter_hashing_workflow(self):
        """Test parameter hashing for cache keys."""
        # Create model configuration
        config = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42}

        # Hash the configuration
        config_hash = hash_params(**config)

        # Verify
        assert isinstance(config_hash, str)
        assert len(config_hash) == 32

        # Same config should produce same hash
        config_hash2 = hash_params(**config)
        assert config_hash == config_hash2

        # Different config should produce different hash
        modified_config = config.copy()
        modified_config["n_estimators"] = 200
        config_hash3 = hash_params(**modified_config)
        assert config_hash != config_hash3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hash_empty_dataframe(self):
        """Test hashing an empty DataFrame."""
        df = pd.DataFrame()
        hash_result = hash_dataframe(df)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_hash_large_dataframe(self):
        """Test hashing a large DataFrame."""
        # Create large DataFrame
        df = pd.DataFrame(np.random.randn(10000, 50))
        hash_result = hash_dataframe(df)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_hash_params_empty(self):
        """Test hashing empty parameters."""
        hash_result = hash_params()

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_hash_params_complex_types(self):
        """Test hashing with complex parameter types."""
        params = {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "test",
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
        }

        hash_result = hash_params(**params)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
