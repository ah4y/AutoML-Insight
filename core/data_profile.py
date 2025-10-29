"""Dataset profiling and meta-feature extraction."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any


class DataProfiler:
    """Extract meta-features from datasets for meta-learning."""
    
    def __init__(self):
        self.profile = {}
    
    def profile_dataset(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Compute comprehensive meta-features for a dataset.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Dictionary of meta-features
        """
        profile = {}
        
        # Basic dimensions
        profile['n_samples'] = X.shape[0]
        profile['n_features'] = X.shape[1]
        profile['dimensionality'] = X.shape[1] / X.shape[0]
        
        # Missing values
        profile['missing_ratio'] = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        profile['features_with_missing'] = X.isnull().any().sum() / X.shape[1]
        
        # Feature types
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        profile['numeric_ratio'] = len(numeric_cols) / X.shape[1]
        profile['categorical_ratio'] = len(categorical_cols) / X.shape[1]
        
        # Detect constant features (zero variance)
        constant_features = X.columns[X.nunique() <= 1].tolist()
        profile['n_constant_features'] = len(constant_features)
        profile['constant_ratio'] = len(constant_features) / X.shape[1]
        if constant_features:
            profile['constant_features'] = constant_features[:10]  # Store first 10
        
        # Statistical properties of numeric features
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # Skewness and kurtosis
            skewness = X_numeric.skew()
            kurtosis = X_numeric.kurtosis()
            profile['mean_skewness'] = skewness.mean()
            profile['mean_kurtosis'] = kurtosis.mean()
            profile['std_skewness'] = skewness.std()
            
            # Correlations (handle NaN from constant features)
            corr_matrix = X_numeric.corr().fillna(0).abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            profile['mean_correlation'] = upper_triangle.stack().mean()
            profile['max_correlation'] = upper_triangle.stack().max()
            
            # PCA-based metrics
            try:
                if X_numeric.shape[0] > X_numeric.shape[1]:
                    pca = PCA(n_components=min(10, X_numeric.shape[1]))
                    pca.fit(X_numeric)
                    profile['pca_95_components'] = np.sum(
                        np.cumsum(pca.explained_variance_ratio_) < 0.95
                    ) + 1
                    profile['pca_first_pc'] = pca.explained_variance_ratio_[0]
            except:
                profile['pca_95_components'] = X.shape[1]
                profile['pca_first_pc'] = 1.0 / X.shape[1]
        else:
            profile['mean_skewness'] = 0.0
            profile['mean_kurtosis'] = 0.0
            profile['std_skewness'] = 0.0
            profile['mean_correlation'] = 0.0
            profile['max_correlation'] = 0.0
            profile['pca_95_components'] = X.shape[1]
            profile['pca_first_pc'] = 0.0
        
        # Target variable statistics (for supervised learning)
        if y is not None:
            if y.dtype in [np.int64, np.int32, object, 'category']:
                # Classification
                profile['task_type'] = 'classification'
                profile['n_classes'] = y.nunique()
                profile['class_imbalance'] = y.value_counts().max() / len(y)
                
                # Class entropy
                class_counts = y.value_counts(normalize=True)
                profile['class_entropy'] = stats.entropy(class_counts)
                
                # Linear separability (simplified)
                if len(numeric_cols) >= 2:
                    profile['linear_separability'] = self._estimate_linear_separability(
                        X[numeric_cols].fillna(X[numeric_cols].median()), y
                    )
                else:
                    profile['linear_separability'] = 0.5
            else:
                # Regression
                profile['task_type'] = 'regression'
                profile['target_skewness'] = stats.skew(y)
                profile['target_kurtosis'] = stats.kurtosis(y)
        else:
            profile['task_type'] = 'clustering'
        
        self.profile = profile
        return profile
    
    def _estimate_linear_separability(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Estimate linear separability using a simple heuristic.
        
        Args:
            X: Numeric features
            y: Target variable
            
        Returns:
            Separability score (0-1)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            
            # Encode target if needed
            if y.dtype == object or y.dtype.name == 'category':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            
            # Quick logistic regression
            lr = LogisticRegression(max_iter=100, random_state=42)
            scores = cross_val_score(lr, X, y_encoded, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.5
    
    def get_profile_vector(self) -> np.ndarray:
        """
        Get meta-features as a vector for meta-learning.
        
        Returns:
            NumPy array of meta-features
        """
        feature_names = [
            'n_samples', 'n_features', 'dimensionality', 'missing_ratio',
            'numeric_ratio', 'categorical_ratio', 'mean_skewness', 'mean_kurtosis',
            'mean_correlation', 'max_correlation', 'pca_95_components', 'pca_first_pc'
        ]
        
        if 'class_entropy' in self.profile:
            feature_names.extend(['n_classes', 'class_imbalance', 'class_entropy', 
                                 'linear_separability'])
        
        vector = [self.profile.get(name, 0.0) for name in feature_names]
        return np.array(vector)
