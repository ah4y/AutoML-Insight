"""Bias-Variance Decomposition Analyzer for Model Evaluation.

This module implements bias-variance decomposition analysis and learning curve
computation to help understand model performance characteristics.
"""

import numpy as np
import logging
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


class BiasVarianceAnalyzer:
    """Analyze bias-variance tradeoff and learning curves for models.
    
    This class provides methods to compute bias-variance decomposition using
    bootstrap resampling and generate learning curves to understand model
    behavior with varying training set sizes.
    
    Attributes:
        n_bootstrap: Number of bootstrap iterations for bias-variance analysis
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, n_bootstrap=50, random_state=42):
        """Initialize the BiasVarianceAnalyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples for decomposition (default: 50)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        logger.info(f"BiasVarianceAnalyzer initialized with {n_bootstrap} bootstrap samples")
    
    def compute_bias_variance_decomposition(self, model, X_train, y_train, X_test, y_test):
        """Compute bias-variance decomposition for a classification model.
        
        Uses bootstrap resampling to estimate bias and variance components of
        the model's prediction error. This helps identify whether the model is
        suffering from high bias (underfitting) or high variance (overfitting).
        
        Args:
            model: Scikit-learn estimator to analyze
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing:
                - bias_squared: Squared bias component
                - variance: Variance component
                - total_error: Total prediction error
                - bias_variance_ratio: Ratio of bias² to variance
                - interpretation: Human-readable interpretation
        """
        try:
            logger.info(f"Computing bias-variance decomposition for {model.__class__.__name__}")
            logger.info(f"Using {self.n_bootstrap} bootstrap samples")
            
            n_test = len(X_test)
            n_classes = len(np.unique(y_train))
            
            # Store predictions from each bootstrap iteration
            predictions = np.zeros((self.n_bootstrap, n_test))
            
            # Bootstrap resampling
            for i in range(self.n_bootstrap):
                if (i + 1) % 10 == 0:
                    logger.info(f"Bootstrap iteration {i + 1}/{self.n_bootstrap}")
                
                # Sample with replacement
                indices = self.rng.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
                y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
                
                # Train model on bootstrap sample
                model_boot = clone(model)
                model_boot.fit(X_boot, y_boot)
                
                # Predict on test set
                predictions[i, :] = model_boot.predict(X_test)
            
            # Compute main prediction (mode across bootstraps)
            main_predictions = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=n_classes).argmax(),
                axis=0,
                arr=predictions
            )
            
            # Convert y_test to numpy array if needed
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            
            # Compute bias² (squared difference between main prediction and true label)
            # For classification, use 0-1 loss
            bias_squared = np.mean(main_predictions != y_test_array)
            
            # Compute variance (average disagreement of bootstrap predictions with main prediction)
            variance = np.mean([
                np.mean(predictions[i, :] != main_predictions)
                for i in range(self.n_bootstrap)
            ])
            
            # Total error (direct 0-1 loss on average predictions)
            total_error = bias_squared + variance
            
            # Bias-variance ratio (bias²/variance)
            bias_variance_ratio = bias_squared / variance if variance > 0 else float('inf')
            
            # Interpretation
            if bias_variance_ratio > 2.0:
                interpretation = "High bias (underfitting) - Model is too simple"
            elif bias_variance_ratio < 0.5:
                interpretation = "High variance (overfitting) - Model is too complex"
            else:
                interpretation = "Good balance between bias and variance"
            
            logger.info(f"Bias-variance decomposition complete: bias²={bias_squared:.4f}, "
                       f"variance={variance:.4f}, ratio={bias_variance_ratio:.2f}")
            
            return {
                'bias_squared': float(bias_squared),
                'variance': float(variance),
                'total_error': float(total_error),
                'bias_variance_ratio': float(bias_variance_ratio),
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Error in bias-variance decomposition: {e}")
            return {
                'bias_squared': None,
                'variance': None,
                'total_error': None,
                'bias_variance_ratio': None,
                'interpretation': f"Analysis failed: {str(e)}"
            }
    
    def compute_learning_curves(self, model, X, y, train_sizes=None, cv=5):
        """Compute learning curves showing performance vs training set size.
        
        Learning curves help diagnose whether a model would benefit from more
        training data. Converging train/test curves suggest the model has learned
        the underlying pattern. Large gaps suggest overfitting.
        
        Args:
            model: Scikit-learn estimator to analyze
            X: Feature matrix (full dataset)
            y: Labels (full dataset)
            train_sizes: List of fractions/absolute sizes for training (default: [0.1, 0.3, 0.5, 0.7, 0.9])
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing:
                - train_sizes: Array of training set sizes used
                - train_scores_mean: Mean training scores
                - train_scores_std: Std dev of training scores
                - test_scores_mean: Mean test scores
                - test_scores_std: Std dev of test scores
        """
        try:
            if train_sizes is None:
                train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            logger.info(f"Computing learning curves for {model.__class__.__name__}")
            logger.info(f"Train sizes: {train_sizes}, CV folds: {cv}")
            
            # Compute learning curves using sklearn
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=model,
                X=X,
                y=y,
                train_sizes=train_sizes,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                shuffle=True
            )
            
            # Compute mean and std
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            logger.info(f"Learning curves computed successfully")
            logger.info(f"Final train score: {train_scores_mean[-1]:.4f} ± {train_scores_std[-1]:.4f}")
            logger.info(f"Final test score: {test_scores_mean[-1]:.4f} ± {test_scores_std[-1]:.4f}")
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores_mean.tolist(),
                'train_scores_std': train_scores_std.tolist(),
                'test_scores_mean': test_scores_mean.tolist(),
                'test_scores_std': test_scores_std.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error computing learning curves: {e}")
            return {
                'train_sizes': [],
                'train_scores_mean': [],
                'train_scores_std': [],
                'test_scores_mean': [],
                'test_scores_std': [],
                'error': str(e)
            }
