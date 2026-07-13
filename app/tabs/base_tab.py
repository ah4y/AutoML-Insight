"""
Base class for dashboard tabs.

Provides common functionality and interface for all tabs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)


class BaseTab(ABC):
    """Base class for all dashboard tabs."""

    def __init__(self):
        """Initialize the tab."""
        self.logger = logger

    @abstractmethod
    def render(self) -> None:
        """
        Render the tab content.

        This method must be implemented by all subclasses.
        """

    def get_session_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from session state.

        Args:
            key: Session state key
            default: Default value if key doesn't exist

        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)

    def set_session_data(self, key: str, value: Any) -> None:
        """
        Set data in session state.

        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value

    def has_data(self) -> bool:
        """Check if data is loaded in session state."""
        return st.session_state.get("data") is not None

    def has_results(self) -> bool:
        """Check if results are available."""
        results = st.session_state.get("results")
        return results is not None and len(results) > 0

    def has_professional_results(self) -> bool:
        """Check if professional AutoML results are available."""
        prof_results = st.session_state.get("professional_results")
        return prof_results is not None and len(prof_results) > 0

    def show_error(self, message: str) -> None:
        """
        Display error message.

        Args:
            message: Error message to display
        """
        st.error(f"❌ {message}")
        self.logger.error(message)

    def show_warning(self, message: str) -> None:
        """
        Display warning message.

        Args:
            message: Warning message to display
        """
        st.warning(f"⚠️ {message}")
        self.logger.warning(message)

    def show_info(self, message: str) -> None:
        """
        Display info message.

        Args:
            message: Info message to display
        """
        st.info(f"ℹ️ {message}")

    def show_success(self, message: str) -> None:
        """
        Display success message.

        Args:
            message: Success message to display
        """
        st.success(f"✅ {message}")

    def require_data(self) -> bool:
        """
        Check if data is loaded, show message if not.

        Returns:
            True if data is available, False otherwise
        """
        if not self.has_data():
            self.show_warning("Please upload or load a dataset first.")
            return False
        return True

    def require_results(self) -> bool:
        """
        Check if results are available, show message if not.

        Returns:
            True if results are available, False otherwise
        """
        if not self.has_results() and not self.has_professional_results():
            self.show_warning("Please run AutoML first to see results.")
            return False
        return True

    def render_metric_card(self, label: str, value: str, delta: Optional[str] = None) -> None:
        """
        Render a metric card.

        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta value
        """
        st.metric(label=label, value=value, delta=delta)

    def render_section_header(self, title: str, description: Optional[str] = None) -> None:
        """
        Render a section header.

        Args:
            title: Section title
            description: Optional description
        """
        st.markdown(f"### {title}")
        if description:
            st.markdown(description)
        st.markdown("---")
