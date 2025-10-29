"""Ensemble methods for combining models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted voting ensemble with calibration.
    Weights are proportional to score / variance.
    """
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        self.weights = weights
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        self.classes_ = np.unique(y)
        
        for model in self.models:
            if not hasattr(model, 'predict_proba'):
                continue
            model.fit(X, y)
        
        # Equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities using weighted average."""
        probas = []
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                probas.append(weight * model.predict_proba(X))
        
        if not probas:
            # Fallback to uniform
            n_classes = len(self.classes_)
            return np.ones((X.shape[0], n_classes)) / n_classes
        
        return np.sum(probas, axis=0)
    
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with meta-learner.
    Base models' predictions are used as features for meta-model.
    """
    
    def __init__(
        self,
        base_models: List[Any],
        meta_model: Any = None,
        use_probas: bool = True
    ):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        self.use_probas = use_probas
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit base models and meta-model."""
        self.classes_ = np.unique(y)
        
        # Fit base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        meta_features = []
        
        for model in self.base_models:
            if self.use_probas and hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                meta_features.append(probas)
            else:
                preds = model.predict(X).reshape(-1, 1)
                meta_features.append(preds)
        
        return np.hstack(meta_features)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict class labels."""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)


class AdaptiveEnsemble:
    """
    Adaptive ensemble that selects best combination of models.
    Uses performance metrics to determine optimal ensemble configuration.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble = None
        self.ensemble_type = None
    
    def create_ensemble(
        self,
        models_dict: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        top_k: int = 3
    ) -> Any:
        """
        Create optimal ensemble from top performing models.
        
        Args:
            models_dict: Dictionary of trained models
            evaluation_results: Model evaluation results
            X: Feature matrix
            y: Target variable
            top_k: Number of top models to include
            
        Returns:
            Ensemble model
        """
        # Get top performing models
        leaderboard = sorted(
            evaluation_results.items(),
            key=lambda x: x[1].get('accuracy_mean', 0),
            reverse=True
        )[:top_k]
        
        if not leaderboard:
            return None
        
        # Extract models and compute weights
        top_models = []
        weights = []
        
        for model_name, results in leaderboard:
            if model_name in models_dict:
                top_models.append(models_dict[model_name])
                
                # Weight = score / (1 + variance)
                score = results.get('accuracy_mean', 0)
                scores_list = results.get('accuracy_scores', [score])
                variance = np.var(scores_list) if len(scores_list) > 1 else 0.01
                
                weight = score / (1 + variance)
                weights.append(weight)
        
        if not top_models:
            return None
        
        # Decide ensemble type based on model diversity
        if len(top_models) >= 3:
            # Use stacking for diverse models
            self.ensemble = StackingEnsemble(
                base_models=top_models,
                use_probas=True
            )
            self.ensemble_type = 'stacking'
        else:
            # Use weighted ensemble for fewer models
            self.ensemble = WeightedEnsemble(
                models=top_models,
                weights=weights
            )
            self.ensemble_type = 'weighted'
        
        # Fit ensemble
        self.ensemble.fit(X, y)
        
        return self.ensemble
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            'type': self.ensemble_type,
            'n_models': len(self.ensemble.models if hasattr(self.ensemble, 'models') 
                           else self.ensemble.base_models),
            'weights': getattr(self.ensemble, 'weights', None)
        }
