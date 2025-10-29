"""Meta-learning model selector."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple


class MetaModelSelector:
    """
    Meta-learning engine to recommend best model families based on dataset characteristics.
    Uses both learned patterns and heuristic rules.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(
        self,
        meta_features_list: List[np.ndarray],
        best_models: List[str]
    ):
        """
        Train meta-model on historical performance data.
        
        Args:
            meta_features_list: List of meta-feature vectors
            best_models: List of best model names for each dataset
        """
        X = np.array(meta_features_list)
        y = best_models
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train meta-model
        self.meta_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state
        )
        self.meta_model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, meta_features: np.ndarray) -> str:
        """
        Predict best model family for a dataset.
        
        Args:
            meta_features: Meta-feature vector
            
        Returns:
            Recommended model name
        """
        if self.is_fitted:
            X_scaled = self.scaler.transform(meta_features.reshape(1, -1))
            return self.meta_model.predict(X_scaled)[0]
        else:
            # Use heuristic rules
            return self._heuristic_selection(meta_features)
    
    def _heuristic_selection(self, meta_features: Dict[str, Any]) -> str:
        """
        Rule-based model selection fallback.
        
        Args:
            meta_features: Dictionary of meta-features
            
        Returns:
            Recommended model name
        """
        # Extract key features
        n_samples = meta_features.get('n_samples', 1000)
        n_features = meta_features.get('n_features', 10)
        dimensionality = meta_features.get('dimensionality', 0.01)
        n_classes = meta_features.get('n_classes', 2)
        class_imbalance = meta_features.get('class_imbalance', 0.5)
        linear_separability = meta_features.get('linear_separability', 0.5)
        
        recommendations = []
        
        # Rule 1: Small dataset with few features → Logistic Regression, KNN
        if n_samples < 1000 and n_features < 20:
            recommendations.append(('LogisticRegression', 0.8))
            recommendations.append(('KNN', 0.7))
        
        # Rule 2: High dimensionality → Tree-based models
        if dimensionality > 0.1 or n_features > 50:
            recommendations.append(('RandomForest', 0.85))
            recommendations.append(('XGBoost', 0.9))
        
        # Rule 3: Linear separability → Linear models, SVM
        if linear_separability > 0.8:
            recommendations.append(('LogisticRegression', 0.9))
            recommendations.append(('LinearSVM', 0.85))
        else:
            recommendations.append(('RBF-SVM', 0.8))
        
        # Rule 4: Large dataset → XGBoost, MLP
        if n_samples > 5000:
            recommendations.append(('XGBoost', 0.9))
            recommendations.append(('MLP', 0.85))
        
        # Rule 5: Multi-class with many classes → Tree ensemble
        if n_classes > 5:
            recommendations.append(('RandomForest', 0.85))
            recommendations.append(('XGBoost', 0.9))
        
        # Rule 6: Imbalanced classes → Ensemble methods
        if class_imbalance > 0.8:
            recommendations.append(('RandomForest', 0.8))
            recommendations.append(('XGBoost', 0.85))
        
        # Aggregate recommendations
        if recommendations:
            model_scores = {}
            for model, score in recommendations:
                if model in model_scores:
                    model_scores[model] = max(model_scores[model], score)
                else:
                    model_scores[model] = score
            
            # Return model with highest score
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
            return best_model
        else:
            # Default recommendation
            return 'XGBoost'
    
    def get_recommendation_with_rationale(
        self,
        meta_features: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get model recommendation with detailed rationale.
        
        Args:
            meta_features: Dataset meta-features
            evaluation_results: Model evaluation results
            
        Returns:
            Dictionary with recommendation and rationale
        """
        # Handle empty evaluation results
        if not evaluation_results:
            return {
                'recommended_model': 'XGBoost',
                'score': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'rationale': ['Default recommendation: No evaluation results available.'],
                'alternatives': []
            }
        
        # Get top performing models from evaluation
        leaderboard = sorted(
            evaluation_results.items(),
            key=lambda x: x[1].get('accuracy_mean', 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )
        
        if not leaderboard:
            return {
                'recommended_model': 'XGBoost',
                'score': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'rationale': ['Default recommendation: XGBoost is versatile and robust.'],
                'alternatives': []
            }
        
        best_model = leaderboard[0][0]
        best_score = leaderboard[0][1].get('accuracy_mean', 0)
        
        # Generate rationale
        rationale = []
        
        # Performance-based
        rationale.append(
            f"{best_model} achieved the highest accuracy: {best_score:.4f}"
        )
        
        # Statistical significance
        if len(leaderboard) > 1:
            second_best = leaderboard[1][0]
            second_score = leaderboard[1][1].get('accuracy_mean', 0)
            diff = best_score - second_score
            
            if diff > 0.01:
                rationale.append(
                    f"Significantly outperforms {second_best} by {diff:.4f}"
                )
        
        # Dataset characteristics
        n_samples = meta_features.get('n_samples', 0)
        n_features = meta_features.get('n_features', 0)
        
        if 'RandomForest' in best_model or 'XGBoost' in best_model:
            rationale.append(
                "Tree-based ensemble well-suited for complex feature interactions"
            )
        
        if 'MLP' in best_model:
            rationale.append(
                "Neural network effective for large datasets with non-linear patterns"
            )
        
        if 'SVM' in best_model:
            rationale.append(
                "Support Vector Machine handles high-dimensional spaces effectively"
            )
        
        # Confidence intervals
        ci_lower = leaderboard[0][1].get('accuracy_ci_lower', best_score)
        ci_upper = leaderboard[0][1].get('accuracy_ci_upper', best_score)
        ci_width = ci_upper - ci_lower
        
        if ci_width < 0.05:
            rationale.append(
                f"Model performance is stable (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
            )
        
        return {
            'recommended_model': best_model,
            'score': best_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'rationale': rationale,
            'alternatives': [
                {
                    'model': model,
                    'score': results.get('accuracy_mean', 0)
                }
                for model, results in leaderboard[1:4]
            ]
        }
