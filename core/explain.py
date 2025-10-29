"""Model explainability using SHAP and other techniques."""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from typing import Any, Optional


class ModelExplainer:
    """Generate explanations for ML models."""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        sample_size: int = 50
    ) -> dict:
        """
        Generate SHAP-based explanations for a model.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            sample_size: Number of samples for SHAP (reduced to 50 to prevent memory issues)
            
        Returns:
            Dictionary containing explanation data
        """
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Sample data if too large
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # For KernelExplainer, use even smaller background dataset
        background_size = min(50, X_sample.shape[0])
        if X_sample.shape[0] > background_size:
            bg_indices = np.random.choice(X_sample.shape[0], background_size, replace=False)
            X_background = X_sample[bg_indices]
        else:
            X_background = X_sample
        
        explanations = {}
        
        # Try to create SHAP explainer
        try:
            # Choose explainer based on model type
            if hasattr(model, 'tree_'):
                # Tree-based models
                self.explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'coef_'):
                # Linear models
                self.explainer = shap.LinearExplainer(model, X_background)
            else:
                # Use KernelExplainer as fallback with small background set
                # KernelExplainer is very memory-intensive, use minimal samples
                if hasattr(model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(model.predict_proba, X_background)
                else:
                    self.explainer = shap.KernelExplainer(model.predict, X_background)
                
                # For KernelExplainer, use even smaller explanation sample
                if X_sample.shape[0] > 20:
                    explain_indices = np.random.choice(X_sample.shape[0], 20, replace=False)
                    X_sample = X_sample[explain_indices]
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Convert to numpy array and handle different formats
            if isinstance(self.shap_values, list):
                # Multi-class case: list of arrays, one per class
                shap_array = np.array([np.abs(sv).mean(axis=0) for sv in self.shap_values])
                shap_importance = shap_array.mean(axis=0)
            else:
                # Convert to array and calculate importance
                shap_array = np.array(self.shap_values)
                if shap_array.ndim == 3:
                    # 3D: (samples, classes, features)
                    shap_importance = np.abs(shap_array).mean(axis=(0, 1))
                elif shap_array.ndim == 2:
                    # 2D: (samples, features)
                    shap_importance = np.abs(shap_array).mean(axis=0)
                elif shap_array.ndim == 1:
                    # 1D: already aggregated
                    shap_importance = np.abs(shap_array)
                else:
                    raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")
            
            # Ensure shap_importance is 1D
            shap_importance = np.atleast_1d(shap_importance).flatten()
            
            # Match length with feature names
            if len(shap_importance) != len(self.feature_names):
                shap_importance = shap_importance[:len(self.feature_names)]
            
            # Convert to Python floats to avoid numpy scalar issues
            explanations['shap_importance'] = {
                name: float(val) for name, val in zip(self.feature_names, shap_importance)
            }
            explanations['shap_values'] = self.shap_values
            explanations['X_sample'] = X_sample
        except Exception as e:
            import traceback
            explanations['shap_error'] = str(e)
            explanations['shap_traceback'] = traceback.format_exc()

        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            explanations['feature_importance'] = {
                name: float(val) for name, val in zip(self.feature_names, model.feature_importances_)
            }
        
        # Coefficient importance (for linear models)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            explanations['coef_importance'] = {
                name: float(val) for name, val in zip(self.feature_names, coef)
            }
        
        return explanations
    
    def plot_shap_summary(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create SHAP summary plot.
        
        Args:
            save_path: Path to save plot
            show: Whether to show plot
        """
        if self.shap_values is None:
            return
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(self.shap_values, list):
            # Multi-class: use first class
            shap.summary_plot(
                self.shap_values[0],
                features=None,
                feature_names=self.feature_names,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values,
                features=None,
                feature_names=self.feature_names,
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_top_features(
        self,
        explanations: dict,
        top_n: int = 10
    ) -> list:
        """
        Get top N most important features.
        
        Args:
            explanations: Explanation dictionary
            top_n: Number of top features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        # Prioritize SHAP importance
        if 'shap_importance' in explanations:
            importance_dict = explanations['shap_importance']
        elif 'feature_importance' in explanations:
            importance_dict = explanations['feature_importance']
        elif 'coef_importance' in explanations:
            importance_dict = explanations['coef_importance']
        else:
            return []
        
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:top_n]
    
    def compute_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42
    ) -> dict:
        """
        Compute permutation importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            random_state: Random seed
            
        Returns:
            Dictionary of feature importances
        """
        try:
            result = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=random_state,
                n_jobs=-1
            )
            
            importance_dict = dict(zip(
                self.feature_names,
                result.importances_mean
            ))
            
            return importance_dict
        except Exception as e:
            return {'error': str(e)}
