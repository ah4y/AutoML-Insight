"""Tests for SessionStateManager."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class _MockSessionState:
    """Instance-based mock for st.session_state that supports all dict-like operations."""

    def __init__(self):
        self._state = {}

    def get(self, key, default=None):
        return self._state.get(key, default)

    def __setitem__(self, key, value):
        self._state[key] = value

    def __getitem__(self, key):
        return self._state[key]

    def __contains__(self, key):
        return key in self._state

    def __delitem__(self, key):
        del self._state[key]

    def __setattr__(self, name, value):
        if name == "_state":
            super().__setattr__(name, value)
        else:
            self._state[name] = value

    def __getattr__(self, name):
        if name == "_state":
            raise AttributeError
        try:
            return self._state[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")

    def keys(self):
        return self._state.keys()

    def clear(self):
        self._state.clear()


class MockStreamlit:
    """Mock streamlit module with instance-based session_state."""

    session_state = _MockSessionState()


# Install mock before importing the module under test
sys.modules["streamlit"] = MockStreamlit()

import numpy as np
import pandas as pd

from app.state.session_manager import SessionStateManager


class TestSessionStateManager:
    """Test suite for SessionStateManager."""

    def setup_method(self):
        """Reset session state before each test."""
        MockStreamlit.session_state.clear()
        SessionStateManager.initialize()

    def test_initialization(self):
        """Test that all defaults are initialized."""
        state = MockStreamlit.session_state._state
        assert "data" in state
        assert "models" in state
        assert "app_stage" in state
        assert state["app_stage"] == "welcome"

    def test_get_set(self):
        """Test basic get/set operations."""
        SessionStateManager.set("test_key", "test_value")
        assert SessionStateManager.get("test_key") == "test_value"
        assert SessionStateManager.get("nonexistent", "default") == "default"

    def test_has(self):
        """Test key existence checking."""
        SessionStateManager.set("exists", True)
        assert SessionStateManager.has("exists")
        assert not SessionStateManager.has("does_not_exist")

    def test_delete(self):
        """Test key deletion."""
        SessionStateManager.set("to_delete", "value")
        assert SessionStateManager.has("to_delete")
        SessionStateManager.delete("to_delete")
        assert not SessionStateManager.has("to_delete")

    def test_get_data(self):
        """Test data getter with validation."""
        # Test with DataFrame
        df = pd.DataFrame({"a": [1, 2, 3]})
        SessionStateManager.set("data", df)
        result = SessionStateManager.get_data()
        assert result is not None
        assert isinstance(result, pd.DataFrame)

        # Test with None
        SessionStateManager.set("data", None)
        assert SessionStateManager.get_data() is None

        # Test with invalid type
        SessionStateManager.set("data", "not a dataframe")
        assert SessionStateManager.get_data() is None

    def test_get_models(self):
        """Test models getter with validation."""
        models = {"model1": object(), "model2": object()}
        SessionStateManager.store_models(models)
        retrieved = SessionStateManager.get_models()
        assert len(retrieved) == 2
        assert "model1" in retrieved

        # Test with invalid type
        SessionStateManager.set("models", "not a dict")
        assert SessionStateManager.get_models() == {}

    def test_store_results(self):
        """Test results storage."""
        prof_results = {"individual_models": {}, "ensemble_models": None}
        SessionStateManager.store_results("professional", prof_results)
        assert SessionStateManager.get_professional_results() == prof_results

        std_results = {"model1": {"accuracy": 0.95}}
        SessionStateManager.store_results("standard", std_results)
        assert SessionStateManager.get_standard_results() == std_results

    def test_clear_results(self):
        """Test results clearing."""
        SessionStateManager.store_results("professional", {"test": "data"})
        SessionStateManager.store_models({"model": object()})
        SessionStateManager.clear_results()

        assert SessionStateManager.get_professional_results() is None
        assert SessionStateManager.get_standard_results() == {}
        assert SessionStateManager.get_models() == {}

    def test_app_stage(self):
        """Test app stage management."""
        SessionStateManager.set_app_stage("configure")
        assert SessionStateManager.get_app_stage() == "configure"

        # Test invalid stage
        SessionStateManager.set_app_stage("invalid_stage")
        assert SessionStateManager.get_app_stage() == "configure"  # Should not change

    def test_random_seed(self):
        """Test random seed management."""
        SessionStateManager.set_random_seed(123)
        assert SessionStateManager.get_random_seed() == 123

        # Test invalid seed
        SessionStateManager.set_random_seed(-1)
        assert SessionStateManager.get_random_seed() == 123  # Should not change

    def test_task_type(self):
        """Test task type management."""
        SessionStateManager.set_task_type("Classification")
        assert SessionStateManager.get_task_type() == "Classification"

        # Test invalid type
        SessionStateManager.set_task_type("InvalidTask")
        assert SessionStateManager.get_task_type() == "Classification"  # Should not change

    def test_has_results(self):
        """Test results existence checking."""
        assert not SessionStateManager.has_results()

        SessionStateManager.store_results("standard", {"test": "data"})
        assert SessionStateManager.has_results()

    def test_get_best_model_name(self):
        """Test best model identification."""
        # Test with professional results
        prof_results = {
            "individual_models": {
                "model1": {"best_score": 0.85},
                "model2": {"best_score": 0.92},
                "model3": {"best_score": 0.78},
            }
        }
        SessionStateManager.store_results("professional", prof_results)
        assert SessionStateManager.get_best_model_name() == "model2"

        # Test with standard results
        SessionStateManager.set("professional_results", None)
        std_results = {
            "model1": {"accuracy": 0.75},
            "model2": {"accuracy": 0.88},
        }
        SessionStateManager.store_results("standard", std_results)
        assert SessionStateManager.get_best_model_name() == "model2"

    def test_optimization_config(self):
        """Test optimization config management."""
        config = SessionStateManager.get_optimization_config()
        assert "time_minutes" in config
        assert config["time_minutes"] == 15

        SessionStateManager.update_optimization_config({"time_minutes": 30})
        config = SessionStateManager.get_optimization_config()
        assert config["time_minutes"] == 30

    def test_processed_data_storage(self):
        """Test processed data storage."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        preprocessor = object()

        SessionStateManager.store_processed_data(X, y, preprocessor)
        X_retrieved, y_retrieved = SessionStateManager.get_processed_data()

        assert np.array_equal(X_retrieved, X)
        assert np.array_equal(y_retrieved, y)
        assert SessionStateManager.get_preprocessor() is preprocessor

    def test_state_summary(self):
        """Test state summary generation."""
        summary = SessionStateManager.get_state_summary()
        assert "has_data" in summary
        assert "app_stage" in summary
        assert "task_type" in summary
        assert summary["has_data"] is False
        assert summary["app_stage"] == "welcome"

    def test_clear_all(self):
        """Test complete state reset."""
        SessionStateManager.set("custom_key", "custom_value")
        SessionStateManager.store_results("professional", {"test": "data"})

        SessionStateManager.clear_all()

        # Should have defaults but not custom keys
        assert SessionStateManager.has("app_stage")
        assert not SessionStateManager.has("custom_key")
        assert SessionStateManager.get_professional_results() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
