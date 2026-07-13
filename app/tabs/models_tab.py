"""
Models Tab - Display standard model results (Classification/Clustering).
"""

import logging

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ModelsTab(BaseTab):
    """Tab for displaying standard model results."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize Models tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the models tab."""
        if not self.require_results():
            return

        task_type = self.get_session_data("task_type", "Classification")

        # If dashboard instance exists, delegate to existing methods
        if self.dashboard is not None:
            if task_type == "Classification":
                self.dashboard.render_classification_results()
            else:
                self.dashboard.render_clustering_results()
            return

        # Otherwise show placeholder
        self._render_placeholder(task_type)

    def _render_placeholder(self, task_type: str) -> None:
        """
        Render placeholder when dashboard not available.

        Args:
            task_type: Type of ML task
        """
        st.subheader(f"🤖 {task_type} Models")
        self.show_info("Model results will be displayed here.")

        results = self.get_session_data("results", {})

        if results:
            st.markdown("### 📊 Results Summary")
            st.json(results)
