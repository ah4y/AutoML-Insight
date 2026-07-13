"""
Advanced Hyperparameter Optimization System
Professional-grade ML engineering approach with Optuna integration
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logger
logger = logging.getLogger(__name__)

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AdvancedHyperparameterOptimizer:
    """
    Professional-grade hyperparameter optimization system using Optuna.
    Implements advanced strategies like multi-objective optimization, early stopping,
    and intelligent search space definition.
    """
    
    def __init__(self, task_type="classification", optimization_time_minutes=10, 
                 n_trials=100, random_state=42, verbose=True):
        """
        Initialize the advanced optimizer.
        
        Args:
            task_type: Type of ML task ('classification', 'regression', 'clustering')
            optimization_time_minutes: Maximum optimization time in minutes
            n_trials: Maximum number of trials
            random_state: Random seed
            verbose: Whether to show progress
        """
        self.task_type = task_type
        self.optimization_time_minutes = optimization_time_minutes
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        
        # Setup sampler and pruner
        self.sampler = TPESampler(seed=random_state, n_startup_trials=10)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        
        # Model-specific search spaces
        self.search_spaces = self._initialize_search_spaces()
        
        # Scoring strategies
        self.scoring_strategies = self._initialize_scoring_strategies()
        
        # Results storage
        self.optimization_results = {}
        self.error_log = []  # Track all errors for debugging
        
    def _initialize_search_spaces(self):
        """Define intelligent search spaces for different model types."""
        return {
            # Tree-based models
            'RandomForestClassifier': {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int_suggest', 3, 20, None),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'max_features': ('categorical', ['sqrt', 'log2', None, 0.5, 0.8]),
                'bootstrap': ('categorical', [True, False])
            },
            'XGBClassifier': {
                'n_estimators': ('int', 100, 1000),
                'max_depth': ('int', 3, 12),
                'learning_rate': ('float', 0.01, 0.3, True),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0.001, 10, True),  # Changed from 0 to 0.001 for log scale
                'reg_lambda': ('float', 0.001, 10, True)  # Changed from 0 to 0.001 for log scale
            },
            'LGBMClassifier': {
                'n_estimators': ('int', 100, 1000),
                'max_depth': ('int', 3, 15),
                'learning_rate': ('float', 0.01, 0.3, True),
                'num_leaves': ('int', 10, 300),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0.001, 10, True),  # Changed from 0 to 0.001 for log scale
                'reg_lambda': ('float', 0.001, 10, True)  # Changed from 0 to 0.001 for log scale
            },
            
            # Neural networks
            'MLPClassifier': {
                'hidden_layer_sizes': ('categorical', [
                    (50,), (100,), (200,), (100, 50), (200, 100), (300, 150, 75)
                ]),
                'activation': ('categorical', ['relu', 'tanh', 'logistic']),
                'solver': ('categorical', ['adam', 'lbfgs', 'sgd']),
                'alpha': ('float', 1e-5, 1e-1, True),
                'learning_rate': ('categorical', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': ('float', 1e-4, 1e-1, True)
            },
            
            # SVM models
            'SVC': {
                'C': ('float', 0.1, 100, True),
                'gamma': ('categorical', ['scale', 'auto'] + [10**i for i in range(-4, 1)]),
                'kernel': ('categorical', ['rbf', 'poly', 'sigmoid']),
                'degree': ('int', 2, 5)  # Only for poly
            },
            
            # Clustering models
            'KMeans': {
                'n_clusters': ('int', 2, 20),
                'init': ('categorical', ['k-means++', 'random']),
                'algorithm': ('categorical', ['lloyd', 'elkan', 'auto']),
                'max_iter': ('int', 100, 500)
            },
            'DBSCAN': {
                'eps': ('float', 0.1, 2.0),
                'min_samples': ('int', 3, 20),
                'algorithm': ('categorical', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            }
        }
    
    def _initialize_scoring_strategies(self):
        """Define scoring strategies for different tasks and scenarios."""
        return {
            'classification': {
                'balanced': 'f1_weighted',
                'precision_focused': 'precision_weighted',
                'recall_focused': 'recall_weighted',
                'accuracy_focused': 'accuracy',
                'multiclass': 'f1_macro'
            },
            'regression': {
                'default': 'neg_mean_squared_error',
                'robust': 'neg_mean_absolute_error',
                'explained_variance': 'r2'
            },
            'clustering': {
                'silhouette': 'silhouette_score',
                'calinski_harabasz': 'calinski_harabasz_score',
                'davies_bouldin': 'davies_bouldin_score'
            }
        }
    
    def optimize_model(self, model, X, y=None, cv_folds=5, scoring_strategy='balanced'):
        """
        Optimize hyperparameters for a single model.
        
        Args:
            model: Sklearn model instance
            X: Feature matrix
            y: Target vector (None for clustering)
            cv_folds: Number of cross-validation folds
            scoring_strategy: Scoring strategy to use
            
        Returns:
            Dictionary with optimization results
        """
        model_name = model.__class__.__name__
        
        # Validate data before optimization
        if self.verbose:
            print(f"\n Validating data for {model_name}...")
            print(f"  X shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
            if y is not None:
                print(f"  y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
                print(f"  y unique values: {len(np.unique(y))}")
        
        if model_name not in self.search_spaces:
            return self._basic_optimization_fallback(model, X, y, cv_folds)
        
        # Create study
        study_name = f"{model_name}_optimization_{int(time.time())}"
        study = optuna.create_study(
            direction='maximize',
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name
        )
        
        # Define objective function
        def objective(trial):
            try:
                return self._objective_function(trial, model, X, y, cv_folds, scoring_strategy, model_name)
            except Exception as e:
                if self.verbose:
                    print(f"Trial failed for {model_name}: {str(e)}")
                raise optuna.TrialPruned()
        
        # Run optimization
        start_time = time.time()
        timeout = self.optimization_time_minutes * 60
        
        try:
            study.optimize(
                objective, 
                n_trials=self.n_trials,
                timeout=timeout,
                show_progress_bar=self.verbose
            )
        except Exception as e:
            if self.verbose:
                print(f"Optimization interrupted: {str(e)}")
        
        # Process results with enhanced error handling
        optimization_time = time.time() - start_time
        
        # Check if any trials completed successfully
        if len(study.trials) == 0:
            raise ValueError("No optimization trials were attempted")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        if self.verbose:
            print(f"   {model_name} Trial Summary:")
            print(f"     Completed: {len(completed_trials)}")
            print(f"     Failed: {len(failed_trials)}")
            print(f"    ✂ Pruned: {len(pruned_trials)}")
        
        if len(completed_trials) == 0:
            # If no trials completed, use the original model
            if self.verbose:
                print(f"Warning: No trials completed successfully for {model_name}. Using original model.")
            baseline_score = self._evaluate_model(model, X, y, cv_folds, scoring_strategy)
            return {
                'model': model,
                'best_params': {},
                'best_score': baseline_score,
                'baseline_score': baseline_score,
                'improvement': 0,
                'improvement_percent': 0,
                'optimization_time': optimization_time,
                'n_trials': len(study.trials),
                'study': study,
                'optimization_failed': True,
                'convergence_info': {
                    'converged': False,
                    'convergence_iteration': None,
                    'final_score': baseline_score
                }
            }
        
        # Check for suspicious results (all trials failed with -1000)
        best_score = study.best_value
        if best_score <= -1000 and len(failed_trials) > len(completed_trials):
            if self.verbose:
                print(f"Warning: {model_name} optimization produced suspicious results (score: {best_score})")
            # Fall back to baseline model
            baseline_score = self._evaluate_model(model, X, y, cv_folds, scoring_strategy)
            return {
                'model': model,
                'best_params': {},
                'best_score': baseline_score,
                'baseline_score': baseline_score,
                'improvement': 0,
                'improvement_percent': 0,
                'optimization_time': optimization_time,
                'n_trials': len(study.trials),
                'study': study,
                'optimization_failed': True,
                'convergence_info': {
                    'converged': False,
                    'convergence_iteration': None,
                    'final_score': baseline_score,
                    'fallback_reason': 'Suspicious optimization results'
                }
            }
        
        # Get best parameters and create optimized model
        best_params = study.best_params
        # Filter out helper parameters that shouldn't be passed to the model
        model_params = {k: v for k, v in best_params.items() if not k.endswith('_use_none')}
        optimized_model = model.__class__(**model_params)
        
        # Fit the optimized model
        if self.task_type == 'clustering':
            optimized_model.fit(X)
        else:
            optimized_model.fit(X, y)
        
        # Calculate improvement
        baseline_score = self._evaluate_model(model, X, y, cv_folds, scoring_strategy)
        optimized_score = study.best_value
        improvement = optimized_score - baseline_score
        
        results = {
            'model': optimized_model,
            'best_params': best_params,
            'best_score': optimized_score,
            'baseline_score': baseline_score,
            'improvement': improvement,
            'improvement_percent': (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0,
            'optimization_time': optimization_time,
            'n_trials': len(study.trials),
            'study': study,
            'convergence_info': {
                'best_trial': study.best_trial.number,
                'trials_to_best': study.best_trial.number + 1
            }
        }
        
        self.optimization_results[model_name] = results
        
        if self.verbose:
            print(f"\n{model_name} Optimization Results:")
            print(f"  Best Score: {optimized_score:.4f}")
            print(f"  Baseline Score: {baseline_score:.4f}")
            print(f"  Improvement: +{improvement:.4f} ({improvement/abs(baseline_score)*100:+.1f}%)")
            print(f"  Time: {optimization_time:.1f}s, Trials: {len(study.trials)}")
        
        return results
    
    def _objective_function(self, trial, base_model, X, y, cv_folds, scoring_strategy, model_name):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            base_model: Base model to optimize
            X: Feature matrix
            y: Target vector
            cv_folds: CV folds
            scoring_strategy: Scoring strategy
            model_name: Name of the model class
            
        Returns:
            Score to maximize
        """
        # Get search space for this model
        search_space = self.search_spaces[model_name]
        
        # Generate parameters for this trial
        params = {}
        for param_name, param_config in search_space.items():
            if param_config[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_config[0] == 'int_suggest':
                # Special case for parameters that can be int or None
                if len(param_config) > 3 and param_config[3] is None:
                    value = trial.suggest_int(param_name, param_config[1], param_config[2])
                    # Randomly decide to use None sometimes
                    if trial.suggest_categorical(f"{param_name}_use_none", [True, False]):
                        params[param_name] = None
                    else:
                        params[param_name] = value
                else:
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_config[0] == 'float':
                log = len(param_config) > 3 and param_config[3]
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=log)
            elif param_config[0] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])
        
        # Handle special cases
        if model_name == 'SVC' and params.get('kernel') != 'poly':
            params.pop('degree', None)
        
        # Filter out helper parameters that shouldn't be passed to the model
        model_params = {k: v for k, v in params.items() if not k.endswith('_use_none')}
        
        # Create model with suggested parameters
        try:
            model = base_model.__class__(**model_params)
        except Exception as e:
            # Log the error for debugging
            if self.verbose:
                print(f"Failed to create model {model_name} with params {model_params}: {str(e)}")
            # Raise an exception to mark this trial as failed, rather than returning a score
            raise optuna.TrialPruned()
        
        # Evaluate model
        try:
            score = self._evaluate_model(model, X, y, cv_folds, scoring_strategy)
            
            # Sanity check for the score
            if score is None or np.isnan(score) or np.isinf(score):
                if self.verbose:
                    print(f"Invalid score for {model_name}: {score}")
                raise optuna.TrialPruned()
            
            # For pruning (early stopping)
            if hasattr(trial, 'report'):
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return score
            
        except optuna.TrialPruned:
            # Re-raise pruning exceptions
            raise
        except Exception as e:
            # Log evaluation errors for debugging
            if self.verbose:
                print(f"Failed to evaluate model {model_name}: {str(e)}")
            # Raise an exception to mark this trial as failed
            raise optuna.TrialPruned()
    
    def _evaluate_model(self, model, X, y, cv_folds, scoring_strategy):
        """
        Evaluate model using appropriate scoring strategy.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv_folds: CV folds
            scoring_strategy: Scoring strategy
            
        Returns:
            Evaluation score
        """
        if self.task_type == 'clustering':
            return self._evaluate_clustering_model(model, X)
        else:
            return self._evaluate_supervised_model(model, X, y, cv_folds, scoring_strategy)
    
    def _evaluate_clustering_model(self, model, X):
        """Evaluate clustering model using silhouette score."""
        try:
            model_copy = model.__class__(**model.get_params())
            labels = model_copy.fit_predict(X)
            
            # Check for valid clustering
            if len(set(labels)) < 2:
                return -1
            
            return silhouette_score(X, labels)
            
        except Exception:
            return -1000
    
    def _evaluate_supervised_model(self, model, X, y, cv_folds, scoring_strategy):
        """Evaluate supervised model using cross-validation."""
        try:
            # Validate input data
            if X is None or y is None:
                logger.error("X or y is None")
                return -1000
            
            # Check for NaN values
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                nan_count = X.isnull().sum().sum()
            else:
                nan_count = np.isnan(X).sum() if hasattr(X, '__len__') else 0
            
            if nan_count > 0:
                logger.error(f"X contains {nan_count} NaN values")
                return -1000
            
            if isinstance(y, pd.Series):
                y_nan_count = y.isnull().sum()
            else:
                y_nan_count = np.isnan(y).sum() if hasattr(y, '__len__') else 0
            
            if y_nan_count > 0:
                logger.error(f"y contains {y_nan_count} NaN values")
                return -1000
            
            # Get scoring metric
            scoring_dict = self.scoring_strategies[self.task_type]
            scoring = scoring_dict.get(scoring_strategy, scoring_dict['balanced'])
            
            # Setup cross-validation
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Manual cross-validation to avoid Python 3.13/Windows joblib issues
            scores = []
            for train_idx, val_idx in cv.split(X, y if self.task_type == 'classification' else None):
                # Split data
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                
                if isinstance(y, pd.Series):
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate score based on metric
                if isinstance(scoring, str):
                    if scoring == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif scoring == 'f1_weighted':
                        score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                    elif scoring == 'neg_mean_squared_error':
                        score = -mean_squared_error(y_val, y_pred)
                    elif scoring == 'r2':
                        score = r2_score(y_val, y_pred)
                    else:
                        # Default to accuracy for classification, r2 for regression
                        if self.task_type == 'classification':
                            score = accuracy_score(y_val, y_pred)
                        else:
                            score = r2_score(y_val, y_pred)
                else:
                    # scoring is a scorer object
                    score = scoring._score_func(y_val, y_pred, **scoring._kwargs)
                
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            error_msg = f"Error evaluating model {model.__class__.__name__}: {str(e)}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            self.error_log.append(tb)
            return -1000
    
    def _basic_optimization_fallback(self, model, X, y, cv_folds):
        """Fallback optimization for models without defined search spaces."""
        
        # Just evaluate the base model
        baseline_score = self._evaluate_model(model, X, y, cv_folds, 'balanced')
        
        return {
            'model': model,
            'best_params': model.get_params(),
            'best_score': baseline_score,
            'baseline_score': baseline_score,
            'improvement': 0,
            'improvement_percent': 0,
            'optimization_time': 0,
            'n_trials': 1,
            'study': None,
            'convergence_info': {'note': 'No search space defined for this model'}
        }
    
    def optimize_ensemble(self, models, X, y, ensemble_type='voting', cv_folds=5):
        """
        Create and optimize ensemble models.
        
        Args:
            models: List of (name, model) tuples
            X: Feature matrix
            y: Target vector
            ensemble_type: Type of ensemble ('voting', 'stacking')
            cv_folds: CV folds
            
        Returns:
            Optimized ensemble model and results
        """
        if self.task_type == 'clustering':
            return None  # Ensembles not applicable for clustering
            
        # First, optimize individual models
        optimized_models = []
        for name, model in models:
            if self.verbose:
                print(f"Optimizing {name}...")
            
            result = self.optimize_model(model, X, y, cv_folds)
            optimized_models.append((name, result['model']))
        
        # Create ensemble
        if ensemble_type == 'voting':
            if self.task_type == 'classification':
                ensemble = VotingClassifier(estimators=optimized_models, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=optimized_models)
        
        elif ensemble_type == 'stacking':
            if self.task_type == 'classification':
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(random_state=self.random_state)
                ensemble = StackingClassifier(
                    estimators=optimized_models,
                    final_estimator=meta_learner,
                    cv=cv_folds
                )
            else:
                from sklearn.linear_model import LinearRegression
                meta_learner = LinearRegression()
                ensemble = StackingRegressor(
                    estimators=optimized_models,
                    final_estimator=meta_learner,
                    cv=cv_folds
                )
        
        # Evaluate ensemble
        ensemble_score = self._evaluate_model(ensemble, X, y, cv_folds, 'balanced')
        
        # Get best individual score for comparison
        individual_scores = [self.optimization_results[name.split('_')[0]]['best_score'] 
                           for name, _ in optimized_models 
                           if name.split('_')[0] in self.optimization_results]
        best_individual_score = max(individual_scores) if individual_scores else 0
        
        ensemble_results = {
            'model': ensemble,
            'ensemble_score': ensemble_score,
            'best_individual_score': best_individual_score,
            'ensemble_improvement': ensemble_score - best_individual_score,
            'individual_results': self.optimization_results
        }
        
        if self.verbose:
            print(f"\nEnsemble Results:")
            print(f"  Ensemble Score: {ensemble_score:.4f}")
            print(f"  Best Individual: {best_individual_score:.4f}")
            print(f"  Ensemble Improvement: +{ensemble_score - best_individual_score:.4f}")
        
        return ensemble_results
    
    def get_optimization_summary(self):
        """Get a summary of all optimization results."""
        if not self.optimization_results:
            return "No optimization results available."
        
        summary = []
        summary.append(" Hyperparameter Optimization Summary")
        summary.append("=" * 50)
        
        total_improvement = 0
        best_model = None
        best_score = -float('inf')
        
        for model_name, results in self.optimization_results.items():
            improvement = results['improvement']
            improvement_percent = results['improvement_percent']
            total_improvement += improvement
            
            if results['best_score'] > best_score:
                best_score = results['best_score']
                best_model = model_name
            
            summary.append(f"\n{model_name}:")
            summary.append(f"   Score: {results['baseline_score']:.4f} → {results['best_score']:.4f}")
            summary.append(f"   Improvement: +{improvement:.4f} ({improvement_percent:+.1f}%)")
            summary.append(f"    Time: {results['optimization_time']:.1f}s")
            summary.append(f"   Trials: {results['n_trials']}")
        
        summary.append(f"\n Best Overall Model: {best_model} (Score: {best_score:.4f})")
        summary.append(f" Total Improvement: +{total_improvement:.4f}")
        
        return "\n".join(summary)

class AutoMLPipeline:
    """
    Professional AutoML pipeline with advanced optimization and model training.
    """
    
    def __init__(self, task_type="classification", optimization_time_minutes=15, random_state=42):
        self.task_type = task_type
        self.optimization_time_minutes = optimization_time_minutes
        self.random_state = random_state
        
        # Initialize optimizer
        self.optimizer = AdvancedHyperparameterOptimizer(
            task_type=task_type,
            optimization_time_minutes=optimization_time_minutes,
            random_state=random_state
        )
        
        # Results storage
        self.pipeline_results = {}
    
    def run_advanced_automl(self, X, y=None, model_candidates=None, include_ensemble=True):
        """
        Run the complete advanced AutoML pipeline with timeout protection.
        
        Args:
            X: Feature matrix
            y: Target vector (None for clustering)
            model_candidates: List of models to optimize
            include_ensemble: Whether to create ensemble models
            
        Returns:
            Complete AutoML results
        """
        import signal
        import time
        
        print(" Starting Advanced AutoML Pipeline...")
        print(f" Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Set up timeout protection (maximum 30 minutes for entire pipeline)
        total_timeout = max(self.optimization_time_minutes * 2, 30) * 60  # At least 30 minutes
        pipeline_start_time = time.time()
        
        # Get default model candidates if none provided
        if model_candidates is None:
            model_candidates = self._get_default_models()
        
        # Individual model optimization
        print(f"\n Optimizing {len(model_candidates)} models...")
        individual_results = {}
        
        for i, (name, model) in enumerate(model_candidates):
            # Check timeout before each model
            elapsed = time.time() - pipeline_start_time
            remaining_time = total_timeout - elapsed
            
            if remaining_time <= 0:
                print(f"\n Pipeline timeout reached. Stopping optimization.")
                break
                
            print(f"\n   Optimizing {name} ({i+1}/{len(model_candidates)})...")
            print(f"   Time remaining: {remaining_time/60:.1f} minutes")
            
            try:
                # Set per-model timeout to remaining time or original limit, whichever is smaller
                model_timeout = min(remaining_time / 60, self.optimization_time_minutes)
                
                # Temporarily adjust optimizer timeout for this model
                original_timeout = self.optimizer.optimization_time_minutes
                self.optimizer.optimization_time_minutes = model_timeout
                
                result = self.optimizer.optimize_model(model, X, y)
                individual_results[name] = result
                
                # Restore original timeout
                self.optimizer.optimization_time_minutes = original_timeout
                
                print(f"   {name} completed in {result.get('optimization_time', 0):.1f}s")
                
            except Exception as e:
                print(f"   {name} optimization failed: {str(e)}")
                # Store failed result
                individual_results[name] = {
                    'model': model,
                    'best_params': {},
                    'best_score': -1000,
                    'optimization_failed': True,
                    'error': str(e)
                }
        
        # Ensemble optimization (only if we have successful individual models)
        ensemble_results = None
        successful_models = [name for name, result in individual_results.items() 
                           if not result.get('optimization_failed', False)]
        
        if include_ensemble and self.task_type != 'clustering' and len(successful_models) > 1:
            elapsed = time.time() - pipeline_start_time
            remaining_time = total_timeout - elapsed
            
            if remaining_time > 60:  # Only try ensemble if we have at least 1 minute left
                print(f"\n Creating ensemble models...")
                try:
                    ensemble_results = self.optimizer.optimize_ensemble(model_candidates, X, y)
                    print(f"   Ensemble optimization completed")
                except Exception as e:
                    print(f"   Ensemble optimization failed: {str(e)}")
                    ensemble_results = None
            else:
                print(f"\n Skipping ensemble optimization - insufficient time remaining")
        
        # Compile final results
        optimization_summary = self.optimizer.get_optimization_summary()
        total_time = time.time() - pipeline_start_time
        
        self.pipeline_results = {
            'individual_models': individual_results,
            'ensemble_models': ensemble_results,
            'optimization_summary': optimization_summary,
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'task_type': self.task_type
            },
            'pipeline_stats': {
                'total_time': total_time,
                'timeout_reached': total_time >= total_timeout,
                'successful_models': len(successful_models),
                'total_models': len(model_candidates)
            }
        }
        
        print(f"\n Advanced AutoML Pipeline Complete!")
        print(f" Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f" Successful models: {len(successful_models)}/{len(model_candidates)}")
        if ensemble_results:
            print(f" Ensemble created: Yes")
        print(optimization_summary)
        
        return self.pipeline_results
    
    def _get_default_models(self):
        """Get default model candidates based on task type."""
        if self.task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            
            return [
                ('RandomForest', RandomForestClassifier(random_state=self.random_state)),
                ('SVM', SVC(random_state=self.random_state, probability=True)),
                ('MLP', MLPClassifier(random_state=self.random_state, max_iter=1000)),
                ('KNN', KNeighborsClassifier())
            ]
        
        elif self.task_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            from sklearn.neighbors import KNeighborsRegressor
            
            return [
                ('RandomForest', RandomForestRegressor(random_state=self.random_state)),
                ('SVM', SVR()),
                ('MLP', MLPRegressor(random_state=self.random_state, max_iter=1000)),
                ('KNN', KNeighborsRegressor())
            ]
        
        else:  # clustering
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.mixture import GaussianMixture
            
            return [
                ('KMeans', KMeans(random_state=self.random_state)),
                ('DBSCAN', DBSCAN()),
                ('GaussianMixture', GaussianMixture(random_state=self.random_state))
            ]
    
    def get_best_model(self):
        """Get the best performing model from the pipeline."""
        if not self.pipeline_results:
            return None
        
        best_score = -float('inf')
        best_model = None
        best_name = None
        
        # Check individual models
        for name, results in self.pipeline_results['individual_models'].items():
            if results['best_score'] > best_score:
                best_score = results['best_score']
                best_model = results['model']
                best_name = name
        
        # Check ensemble models
        if self.pipeline_results['ensemble_models']:
            ensemble_score = self.pipeline_results['ensemble_models']['ensemble_score']
            if ensemble_score > best_score:
                best_score = ensemble_score
                best_model = self.pipeline_results['ensemble_models']['model']
                best_name = "Ensemble"
        
        return {
            'name': best_name,
            'model': best_model,
            'score': best_score
        }