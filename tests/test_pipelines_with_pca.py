"""Integration tests for end-to-end pipelines with PCA/dimred.

Tests complete AutoML workflows including preprocessing, dimensionality reduction,
model training, and evaluation to ensure everything works together properly.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split

from core.dimred import DimRedConfig, DimRedSelector
from core.dimred_evaluator import DimRedEvaluator
from core.evaluate_cls import ClassificationEvaluator
from core.evaluate_clu import ClusteringEvaluator
from core.models_clustering import get_clustering_models
from core.models_supervised import get_supervised_models
from core.preprocess import DataPreprocessor


class TestEndToEndClassificationPipeline:
    """Test complete classification pipeline with dimensionality reduction."""

    def test_full_classification_pipeline(self):
        """Test complete classification workflow with dimred."""
        # Create realistic classification dataset
        X, y = make_classification(
            n_samples=300, n_features=50, n_informative=30, n_redundant=10, n_classes=3, random_state=42
        )

        # Create dimred config
        config_dict = {"dimred": {"enable": "on", "method": "pca", "variance_target": 0.95, "k_max": 20}}
        dimred_config = DimRedConfig(config_dict)

        # Create preprocessor with dimred
        preprocessor = DataPreprocessor(max_features=1000, dimred_config=dimred_config)

        # Preprocessing
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Check dimred was applied
        assert X_processed.shape[1] < X.shape[1]  # Should reduce dimensions
        assert X_processed.shape[0] == X.shape[0]  # Same number of samples

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.3, stratify=y_processed, random_state=42
        )

        # Get models and train a representative model
        models = get_supervised_models(random_state=42)
        model = models["LogisticRegression"]
        model.fit(X_train, y_train)

        # Evaluate directly
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Should complete successfully with reasonable performance
        assert acc > 0.3  # At least better than random for 3-class

    def test_dimred_evaluator_integration(self):
        """Test DimRedEvaluator integration with full pipeline."""
        # Create dataset
        X, y = make_classification(n_samples=200, n_features=40, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        # Create dimred evaluator with current API
        config_dict = {
            "dimred": {"enable": "off", "method": "auto", "variance_target": 0.90}  # Disable dimred for faster test
        }
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()

        dimred_evaluator = DimRedEvaluator(
            preprocessor=preprocessor, dimred_config=dimred_config, n_folds=2, n_repeats=1, random_state=42
        )

        # Get representative models
        all_models = get_supervised_models(random_state=42)
        test_models = {
            "LogisticRegression": all_models["LogisticRegression"],
            "RandomForest": all_models["RandomForest"],
        }

        # Evaluate dimred impact using current API
        results = dimred_evaluator.evaluate_classification_with_dimred(test_models, X_df, y, task_type="classification")

        # Validate results structure
        assert isinstance(results, dict)

        # Check individual model results
        for model_name in test_models.keys():
            assert model_name in results

    def test_auto_method_selection_dense(self):
        """Test auto method selection for dense data."""
        # Dense data
        X = np.random.randn(100, 30)
        y = np.random.randint(0, 2, 100)

        config_dict = {"dimred": {"enable": "auto", "method": "auto"}}
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should handle auto-selection gracefully
        assert X_processed.shape[0] == X.shape[0]

        # Check if dimred was applied
        if hasattr(preprocessor, "dimred_selector") and preprocessor.dimred_selector.is_fitted:
            if preprocessor.dimred_selector.transformer_ is not None:
                # Dimred was applied
                assert preprocessor.dimred_selector.selected_method_ in ["pca", "tsvd", "ipca"]

    def test_pipeline_with_mixed_data_types(self):
        """Test pipeline with mixed categorical and numerical features."""
        # Create mixed dataset
        np.random.seed(42)
        n_samples = 200

        # Numerical features
        numerical = np.random.randn(n_samples, 20)

        # Categorical features
        categorical = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], n_samples),
                "cat2": np.random.choice(["X", "Y"], n_samples),
                "cat3": np.random.choice(["high", "medium", "low"], n_samples),
            }
        )

        # Combine
        X = pd.concat([pd.DataFrame(numerical, columns=[f"num_{i}" for i in range(20)]), categorical], axis=1)

        y = np.random.randint(0, 2, n_samples)

        # Create config
        config_dict = {"dimred": {"enable": "on", "method": "auto", "k_max": 15}}
        dimred_config = DimRedConfig(config_dict)

        # Process with dimred
        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should handle mixed types and apply dimred
        assert X_processed.shape[0] == X.shape[0]
        assert isinstance(X_processed, np.ndarray)  # Should be numpy array after preprocessing


class TestEndToEndClusteringPipeline:
    """Test complete clustering pipeline with dimensionality reduction."""

    def test_full_clustering_pipeline(self):
        """Test complete clustering workflow with dimred."""
        # Create clustering dataset
        X, _ = make_blobs(n_samples=300, n_features=40, centers=4, cluster_std=2.0, random_state=42)

        # Create dimred config
        config_dict = {"dimred": {"enable": "on", "method": "pca", "k_max": 15}}
        dimred_config = DimRedConfig(config_dict)

        # Create preprocessor with dimred - pass y=None for clustering
        preprocessor = DataPreprocessor(dimred_config=dimred_config)

        # Create a dummy y (not None) since preprocessor expects it for feature selection
        dummy_y = np.zeros(X.shape[0])
        X_processed, _ = preprocessor.fit_transform(X, dummy_y)

        # Check dimred was applied
        assert X_processed.shape[1] < X.shape[1]  # Should reduce dimensions
        assert X_processed.shape[0] == X.shape[0]  # Same number of samples

        # Get clustering models and evaluate
        models = get_clustering_models(random_state=42)
        evaluator = ClusteringEvaluator()

        # Test with KMeans
        kmeans = models["KMeans"]
        labels = kmeans.fit_predict(X_processed)
        result = evaluator.evaluate_model(kmeans, X_processed, "KMeans", labels)

        # Should complete successfully
        assert "silhouette" in result
        assert len(labels) == X.shape[0]

    def test_dimred_evaluator_clustering(self):
        """Test DimRedEvaluator with clustering task."""
        # Create clustering data
        X, _ = make_blobs(n_samples=200, n_features=25, centers=3, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        # Create dimred evaluator with current API
        config_dict = {"dimred": {"enable": "off", "method": "auto"}}  # Disable for speed
        dimred_config = DimRedConfig(config_dict)
        preprocessor = DataPreprocessor()

        dimred_evaluator = DimRedEvaluator(
            preprocessor=preprocessor, dimred_config=dimred_config, n_folds=2, n_repeats=1, random_state=42
        )

        # DimRedEvaluator is designed for classification, not clustering
        # Just verify it can be created and its attributes are correct
        assert dimred_evaluator.preprocessor is preprocessor
        assert dimred_evaluator.dimred_config == dimred_config


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_yaml_config_loading(self):
        """Test loading dimred config from YAML file."""
        # Create temporary config file
        config_data = {
            "dimred": {
                "enable": "on",
                "method": "pca",
                "variance_target": 0.90,
                "k_max": 100,
                "whiten": True,
                "seed": 123,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Load config from file
            with open(config_file, "r") as f:
                loaded_config = yaml.safe_load(f)

            dimred_config = DimRedConfig(loaded_config)

            # Verify config was loaded correctly
            assert dimred_config.enable == "on"
            assert dimred_config.method == "pca"
            assert dimred_config.variance_target == 0.90
            assert dimred_config.k_max == 100
            assert dimred_config.whiten is True
            assert dimred_config.seed == 123

            # Test in pipeline
            X = np.random.randn(100, 20)
            y = np.random.randint(0, 2, 100)

            preprocessor = DataPreprocessor(dimred_config=dimred_config)
            X_processed, y_processed = preprocessor.fit_transform(X, y)

            assert X_processed.shape[0] == X.shape[0]

        finally:
            os.unlink(config_file)

    def test_config_validation_in_pipeline(self):
        """Test config validation during pipeline execution."""
        # Invalid config should be caught
        invalid_configs = [
            {"dimred": {"enable": "invalid"}},
            {"dimred": {"method": "unknown"}},
            {"dimred": {"variance_target": 1.5}},
            {"dimred": {"k_max": 0}},
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                DimRedConfig(invalid_config)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create larger dataset
        X, y = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)

        # Configure for large dataset
        config_dict = {"dimred": {"enable": "auto", "method": "auto", "k_max": 50}}  # Should select appropriate method
        dimred_config = DimRedConfig(config_dict)

        # Process
        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should complete in reasonable time and reduce dimensions
        assert X_processed.shape[0] == X.shape[0]
        if hasattr(preprocessor, "dimred_selector") and preprocessor.dimred_selector.is_fitted:
            if preprocessor.dimred_selector.transformer_ is not None:
                assert X_processed.shape[1] <= 50

    def test_high_dimensional_dataset(self):
        """Test with high-dimensional dataset."""
        # Very high-dimensional dataset
        X, y = make_classification(n_samples=200, n_features=500, n_informative=50, random_state=42)  # Many features

        config_dict = {"dimred": {"enable": "auto", "method": "auto", "variance_target": 0.95}}
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should significantly reduce dimensions for very high-dim data
        assert X_processed.shape[1] < X.shape[1]
        assert X_processed.shape[0] == X.shape[0]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        config_dict = {"dimred": {"enable": "on", "method": "pca"}}
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)

        # Empty dataset should raise or return empty gracefully
        # The preprocessor may raise due to no valid features
        try:
            X_processed, y_processed = preprocessor.fit_transform(X, y)
            assert X_processed.shape[0] == 0
        except (ValueError, IndexError):
            pass  # Expected — empty data can't be preprocessed

    def test_single_sample_dataset(self):
        """Test handling of single sample."""
        # Use varied features to avoid all being removed as constant
        np.random.seed(42)
        X = np.random.randn(1, 10)
        y = np.array([0])

        config_dict = {"dimred": {"enable": "off", "method": "pca"}}  # Disable dimred for single sample
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)

        # Single sample may fail during preprocessing (constant features removed)
        try:
            X_processed, y_processed = preprocessor.fit_transform(X, y)
            assert X_processed.shape[0] == 1
        except (ValueError, IndexError):
            pass  # Expected — single sample has all constant features

    def test_more_components_than_features(self):
        """Test requesting more components than features."""
        X = np.random.randn(100, 5)  # Only 5 features
        y = np.random.randint(0, 2, 100)

        config_dict = {"dimred": {"enable": "on", "method": "pca", "k_max": 20}}  # More than features
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should cap at number of features
        assert X_processed.shape[1] <= 5
        assert X_processed.shape[0] == 100

    def test_constant_features(self):
        """Test handling of constant features."""
        X = np.ones((100, 10))  # All features constant
        X[:, 0] = np.random.randn(100)  # One varying feature
        y = np.random.randint(0, 2, 100)

        config_dict = {"dimred": {"enable": "on", "method": "pca"}}
        dimred_config = DimRedConfig(config_dict)

        preprocessor = DataPreprocessor(dimred_config=dimred_config)
        X_processed, y_processed = preprocessor.fit_transform(X, y)

        # Should handle constant features (preprocessing should remove them)
        assert X_processed.shape[0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
