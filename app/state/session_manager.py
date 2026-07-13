"""Centralized session state management for Streamlit app."""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class SessionStateManager:
    """
    Centralized manager for Streamlit session state.

    Provides type-safe accessors, validation, and cleanup for session state variables.
    All session state modifications should go through this manager to ensure consistency.
    """

    # Default values for all session state variables
    DEFAULTS = {
        "data": None,
        "results": {},
        "professional_results": None,
        "models": {},
        "profiler": None,
        "ai_engine": None,
        "enhanced_ai_engine": None,
        "ai_insights": None,
        "execution_mode": "local",
        "jupyter_server_url": "",
        "jupyter_token": "",
        "jupyter_connected": False,
        "remote_logs": [],
        "show_configuration": False,
        "config_analyzed": False,
        "dataset_config": {},
        "optimization_config": {
            "time_minutes": 15,
            "max_trials": 100,
            "include_ensemble": True,
            "advanced_features": [],
        },
        "app_stage": "welcome",
        "show_feature_engineering": False,
        "feature_engineering_applied": False,
        "selected_columns": None,
        "ai_analysis": None,
        "dimred_enabled": "auto",
        "dimred_method": "auto",
        "dimred_variance_target": 0.95,
        "dimred_k_max": 256,
        "dimred_config": None,
        "dimred_results": None,
        "enable_class_filter": False,
        "min_class_samples": 5,
        "random_seed": 42,
        "X_processed": None,
        "y_processed": None,
        "preprocessor": None,
        "task_performed": None,
        "task_type": None,
        "target_col": None,
    }

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables with defaults."""
        for key, value in SessionStateManager.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug(f"Initialized session state: {key}")

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Safely get a value from session state.

        Args:
            key: Session state key
            default: Default value if key doesn't exist

        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state with validation.

        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
        logger.debug(f"Updated session state: {key}")

    @staticmethod
    def has(key: str) -> bool:
        """Check if a key exists in session state."""
        return key in st.session_state

    @staticmethod
    def delete(key: str) -> None:
        """
        Delete a key from session state.

        Args:
            key: Session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
            logger.debug(f"Deleted session state: {key}")

    # Specialized getters with validation

    @staticmethod
    def get_data() -> Optional[pd.DataFrame]:
        """Get the current dataset."""
        data = st.session_state.get("data")
        if data is not None and not isinstance(data, pd.DataFrame):
            logger.error(f"Data is not a DataFrame: {type(data)}")
            return None
        return data

    @staticmethod
    def get_models() -> Dict[str, Any]:
        """Get trained models dictionary."""
        models = st.session_state.get("models", {})
        if not isinstance(models, dict):
            logger.warning(f"Models is not a dict: {type(models)}, returning empty dict")
            return {}
        return models

    @staticmethod
    def get_professional_results() -> Optional[Dict]:
        """Get professional AutoML results."""
        return st.session_state.get("professional_results")

    @staticmethod
    def get_standard_results() -> Dict:
        """Get standard AutoML results."""
        return st.session_state.get("results", {})

    @staticmethod
    def get_best_model_name() -> Optional[str]:
        """Get the name of the best performing model."""
        # Check professional results first
        prof_results = SessionStateManager.get_professional_results()
        if prof_results and "individual_models" in prof_results:
            best_score = -float("inf")
            best_name = None
            for name, result in prof_results["individual_models"].items():
                score = result.get("best_score", -float("inf"))
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name:
                return best_name

        # Fall back to standard results
        results = SessionStateManager.get_standard_results()
        if results:
            best_score = -float("inf")
            best_name = None
            for name, result in results.items():
                if isinstance(result, dict):
                    score = result.get("accuracy", result.get("silhouette", -float("inf")))
                    if score > best_score:
                        best_score = score
                        best_name = name
            return best_name

        return None

    @staticmethod
    def get_best_model() -> Optional[Any]:
        """Get the best performing model object."""
        best_name = SessionStateManager.get_best_model_name()
        if best_name:
            models = SessionStateManager.get_models()
            return models.get(best_name)
        return None

    @staticmethod
    def store_results(results_type: str, data: Dict) -> None:
        """
        Store AutoML results with validation.

        Args:
            results_type: 'professional' or 'standard'
            data: Results dictionary
        """
        if not isinstance(data, dict):
            logger.error(f"Results data must be dict, got {type(data)}")
            return

        if results_type == "professional":
            st.session_state.professional_results = data
            logger.info("Stored professional AutoML results")
        elif results_type == "standard":
            st.session_state.results = data
            logger.info("Stored standard AutoML results")
        else:
            logger.warning(f"Unknown results type: {results_type}")

    @staticmethod
    def store_models(models: Dict[str, Any]) -> None:
        """
        Store trained models with validation.

        Args:
            models: Dictionary of model_name -> model_object
        """
        if not isinstance(models, dict):
            logger.error(f"Models must be dict, got {type(models)}")
            return

        st.session_state.models = models
        logger.info(f"Stored {len(models)} trained models")

    @staticmethod
    def clear_results() -> None:
        """Clear all results and models."""
        st.session_state.results = {}
        st.session_state.professional_results = None
        st.session_state.models = {}
        logger.info("Cleared all results and models")

    @staticmethod
    def clear_all() -> None:
        """Reset all session state to defaults."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionStateManager.initialize()
        logger.info("Reset all session state to defaults")

    @staticmethod
    def get_app_stage() -> str:
        """Get current app stage (welcome, configure, results)."""
        return st.session_state.get("app_stage", "welcome")

    @staticmethod
    def set_app_stage(stage: str) -> None:
        """Set app stage with validation."""
        valid_stages = ["welcome", "configure", "results"]
        if stage not in valid_stages:
            logger.warning(f"Invalid app stage: {stage}, must be one of {valid_stages}")
            return

        st.session_state.app_stage = stage
        logger.info(f"App stage changed to: {stage}")

    @staticmethod
    def get_random_seed() -> int:
        """Get the random seed for reproducibility."""
        return st.session_state.get("random_seed", 42)

    @staticmethod
    def set_random_seed(seed: int) -> None:
        """Set random seed with validation."""
        if not isinstance(seed, int) or seed < 0:
            logger.warning(f"Invalid random seed: {seed}, must be non-negative integer")
            return

        st.session_state.random_seed = seed
        logger.info(f"Random seed set to: {seed}")

    @staticmethod
    def get_task_type() -> Optional[str]:
        """Get the ML task type (Classification, Regression, Clustering)."""
        return st.session_state.get("task_type")

    @staticmethod
    def set_task_type(task_type: str) -> None:
        """Set task type with validation."""
        valid_types = ["Classification", "Regression", "Clustering"]
        if task_type not in valid_types:
            logger.warning(f"Invalid task type: {task_type}, must be one of {valid_types}")
            return

        st.session_state.task_type = task_type
        logger.info(f"Task type set to: {task_type}")

    @staticmethod
    def get_optimization_config() -> Dict:
        """Get hyperparameter optimization configuration."""
        return st.session_state.get(
            "optimization_config",
            {"time_minutes": 15, "max_trials": 100, "include_ensemble": True, "advanced_features": []},
        )

    @staticmethod
    def update_optimization_config(updates: Dict) -> None:
        """Update optimization configuration."""
        config = SessionStateManager.get_optimization_config()
        config.update(updates)
        st.session_state.optimization_config = config
        logger.info(f"Updated optimization config: {updates}")

    @staticmethod
    def is_ai_enabled() -> bool:
        """Check if AI features are available."""
        return st.session_state.get("ai_engine") not in [None, False]

    @staticmethod
    def has_results() -> bool:
        """Check if any AutoML results exist."""
        return bool(st.session_state.get("results")) or bool(st.session_state.get("professional_results"))

    @staticmethod
    def get_preprocessor() -> Optional[Any]:
        """Get the data preprocessor."""
        return st.session_state.get("preprocessor")

    @staticmethod
    def get_processed_data() -> tuple:
        """Get processed X and y data."""
        X = st.session_state.get("X_processed")
        y = st.session_state.get("y_processed")
        return X, y

    @staticmethod
    def store_processed_data(X: Any, y: Any, preprocessor: Any) -> None:
        """Store processed data and preprocessor."""
        st.session_state.X_processed = X
        st.session_state.y_processed = y
        st.session_state.preprocessor = preprocessor
        logger.info("Stored processed data and preprocessor")

    @staticmethod
    def get_state_summary() -> Dict[str, Any]:
        """Get a summary of current session state for debugging."""
        summary = {
            "has_data": SessionStateManager.get_data() is not None,
            "data_shape": SessionStateManager.get_data().shape if SessionStateManager.get_data() is not None else None,
            "has_results": SessionStateManager.has_results(),
            "num_models": len(SessionStateManager.get_models()),
            "best_model": SessionStateManager.get_best_model_name(),
            "app_stage": SessionStateManager.get_app_stage(),
            "task_type": SessionStateManager.get_task_type(),
            "ai_enabled": SessionStateManager.is_ai_enabled(),
            "random_seed": SessionStateManager.get_random_seed(),
        }
        return summary

    @staticmethod
    def log_state_summary() -> None:
        """Log current session state summary."""
        summary = SessionStateManager.get_state_summary()
        logger.info(f"Session State Summary: {summary}")
