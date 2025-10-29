"""Hyperparameter tuning with Optuna."""

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
from typing import Any, Dict


class OptunaHyperparameterTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(
        self,
        n_trials: int = 20,
        cv: int = 3,
        scoring: str = 'accuracy',
        random_state: int = 42,
        verbose: bool = False
    ):
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.best_params = {}
        self.best_score = None
    
    def tune(self, model_name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Tune hyperparameters for a given model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuned model instance
        """
        def objective(trial):
            params = self._get_param_space(trial, model_name)
            tuned_model = clone(model)
            
            # Set parameters
            for param, value in params.items():
                setattr(tuned_model, param, value)
            
            # Cross-validation
            try:
                scores = cross_val_score(
                    tuned_model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                return 0.0
        
        # Run optimization
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.verbose else optuna.logging.WARNING
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Create tuned model
        tuned_model = clone(model)
        for param, value in self.best_params.items():
            setattr(tuned_model, param, value)
        
        # Fit on full data
        tuned_model.fit(X, y)
        
        return tuned_model
    
    def _get_param_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Define parameter search space for different models."""
        if model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        elif model_name == 'RBF-SVM':
            return {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
        elif model_name == 'KNN':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            }
        elif model_name == 'MLP':
            return {
                'hidden_layers': trial.suggest_categorical(
                    'hidden_layers',
                    [(64,), (128,), (64, 32), (128, 64), (256, 128)]
                ),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            }
        elif model_name == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 0.001, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
            }
        else:
            return {}
