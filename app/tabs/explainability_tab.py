"""
Explainability Tab - Display SHAP values and feature importance.
"""

import logging

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ExplainabilityTab(BaseTab):
    """Tab for displaying model explainability results."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize Explainability tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the explainability tab."""
        # If dashboard instance exists, delegate to existing method
        if self.dashboard is not None:
            self.dashboard.render_explainability()
            return

        # Otherwise show simplified version
        self._render_simplified()

    def _render_simplified(self) -> None:
        """Render simplified explainability analysis."""
        st.subheader("🔍 Model Explainability")

        if not self.require_results():
            return

        results = self.get_session_data("results", {})

        if results:
            st.markdown("### 🎯 Feature Importance")
            self.show_info("Explainability analysis will be displayed here.")

            # Placeholder for SHAP values
            with st.expander("📊 **SHAP Values**"):
                st.info("SHAP analysis is computationally intensive. It will be computed on demand.")
        else:
            self.show_warning("No results available for explainability analysis.")
