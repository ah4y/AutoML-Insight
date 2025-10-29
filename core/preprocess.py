"""Data preprocessing pipeline with two-stage feature selection."""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from typing import Tuple, Optional, List, Any
import logging
from utils.logging_utils import setup_logger


class DataPreprocessor:
    """
    Robust preprocessing pipeline for ML datasets.
    
    Handles missing values, feature scaling, encoding, and intelligent
    feature selection to prevent memory issues with high-dimensional data.
    
    Attributes:
        preprocessor (Optional[ColumnTransformer]): Fitted sklearn preprocessor
        numeric_features (List[str]): List of numeric feature names
        categorical_features (List[str]): List of categorical feature names
        feature_names (List[str]): List of transformed feature names
        logger (logging.Logger): Logger instance
        max_features (int): Maximum number of features to retain
        feature_selector (Optional[SelectKBest]): Fitted feature selector
        label_encoder (Optional[LabelEncoder]): Fitted label encoder for target
    """
    
    def __init__(self, max_features: int = 1000) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            max_features: Maximum number of features to keep (default: 1000)
        """
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names: List[str] = []
        self.logger: logging.Logger = setup_logger()
        self.max_features: int = max_features
        self.feature_selector: Optional[SelectKBest] = None
        self.label_encoder: Optional[LabelEncoder] = None
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data with two-stage feature selection.
        
        Stage 1: Remove constant/low-variance features before transformation
        Stage 2: Apply SelectKBest after transformation if needed
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,), optional for unsupervised tasks
            
        Returns:
            Tuple containing:
                - X_transformed: Preprocessed features as numpy array
                - y_transformed: Label-encoded target as numpy array (None if y not provided)
                
        Raises:
            ValueError: If X is empty or has no valid features after preprocessing
            
        Example:
            >>> preprocessor = DataPreprocessor(max_features=100)
            >>> X_train_transformed, y_train_transformed = preprocessor.fit_transform(X_train, y_train)
            >>> print(X_train_transformed.shape)
            (150, 100)
        """
        # Remove constant features first (zero variance)
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            self.logger.warning(f"Removing {len(constant_features)} constant features (zero variance)")
            if len(constant_features) <= 5:
                self.logger.info(f"Constant features: {constant_features}")
            X = X.drop(columns=constant_features)
        
        # Remove low-variance features (quasi-constant)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Calculate variance for numeric columns
            variances = X[numeric_cols].var()
            low_var_features = variances[variances < 0.01].index.tolist()
            if low_var_features:
                self.logger.warning(f"Removing {len(low_var_features)} low-variance features (var < 0.01)")
                X = X.drop(columns=low_var_features)
        
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # PRE-SELECTION: If too many features, select before transformation to avoid memory issues
        if len(self.numeric_features) > self.max_features and y is not None:
            self.logger.warning(f"Dataset has {len(self.numeric_features):,} numeric features. Pre-selecting top {self.max_features:,} before transformation to prevent memory issues...")
            
            # Use variance threshold first for quick reduction
            X_numeric = X[self.numeric_features].fillna(X[self.numeric_features].median())
            
            # Calculate correlation with target for feature ranking
            from sklearn.preprocessing import LabelEncoder
            if not pd.api.types.is_numeric_dtype(y):
                y_numeric = LabelEncoder().fit_transform(y)
            else:
                y_numeric = y
            
            # Calculate absolute correlation with target
            correlations = X_numeric.corrwith(pd.Series(y_numeric)).abs()
            correlations = correlations.fillna(0)  # Handle NaN correlations
            
            # Select top features by correlation
            top_features = correlations.nlargest(self.max_features).index.tolist()
            self.logger.info(f"Pre-selected {len(top_features):,} features based on correlation with target")
            
            # Keep only selected numeric features
            self.numeric_features = top_features
            X = X[self.numeric_features + self.categorical_features]
        
        # Numeric pipeline: impute + scale
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute + one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_pipeline, self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_pipeline, self.categorical_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Extract feature names
        self.feature_names = self._get_feature_names()
        
        # Apply feature selection if we have too many features
        if X_transformed.shape[1] > self.max_features and y is not None:
            self.logger.warning(f"Dataset has {X_transformed.shape[1]:,} features. Selecting top {self.max_features:,} features to prevent memory issues...")
            
            # Use SelectKBest to select most important features
            if pd.api.types.is_numeric_dtype(y) or (hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number)):
                # Check if it's continuous (regression) or discrete (classification)
                n_unique = len(np.unique(y))
                if n_unique > 20:  # Likely regression
                    score_func = f_regression
                else:  # Classification
                    score_func = f_classif
            else:
                score_func = f_classif
            
            self.feature_selector = SelectKBest(score_func=score_func, k=min(self.max_features, X_transformed.shape[1]))
            X_transformed = self.feature_selector.fit_transform(X_transformed, y)
            
            # Update feature names to only selected features
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Feature selection complete. Reduced from {len(selected_indices):,} to {len(self.feature_names):,} features")
        
        # Handle target variable
        y_transformed = None
        if y is not None:
            # Always use LabelEncoder to ensure 0-indexed contiguous class labels
            # This works for both string labels and numeric class labels
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y_transformed = self.label_encoder.fit_transform(y)
            
            # Log the label mapping for transparency
            self.logger.info(f"Label encoding applied: {len(self.label_encoder.classes_)} classes")
            if len(self.label_encoder.classes_) <= 20:
                # Show mapping if reasonable number of classes
                class_mapping = {orig: new for new, orig in enumerate(self.label_encoder.classes_)}
                self.logger.info(f"Class mapping: {class_mapping}")
        
        return X_transformed, y_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Applies the same preprocessing pipeline fitted during fit_transform,
        including feature selection if it was used during training.
        
        Args:
            X: Feature matrix to transform (n_samples, n_features)
            
        Returns:
            Transformed features as numpy array (n_samples, n_selected_features)
            
        Raises:
            ValueError: If preprocessor not fitted (call fit_transform first)
            
        Example:
            >>> X_test_transformed = preprocessor.transform(X_test)
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X_transformed = self.preprocessor.transform(X)
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        return X_transformed
    
    def _get_feature_names(self) -> List[str]:
        """
        Extract feature names after transformation.
        
        Combines numeric feature names with one-hot encoded categorical
        feature names to create complete list of transformed features.
        
        Returns:
            List of feature names after preprocessing
        """
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Get one-hot encoded feature names
                try:
                    ohe = transformer.named_steps['onehot']
                    cat_features = ohe.get_feature_names_out(features)
                    feature_names.extend(cat_features)
                except:
                    feature_names.extend(features)
        
        return feature_names
    
    def get_feature_names(self) -> List[str]:
        """
        Get transformed feature names.
        
        Returns:
            List of feature names after preprocessing and selection
        """
        return self.feature_names
