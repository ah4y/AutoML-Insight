"""
Professional AutoML Tab - Display advanced optimization results.
"""

import logging

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ProfessionalAutoMLTab(BaseTab):
    """Tab for displaying professional AutoML results."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize Professional AutoML tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the professional AutoML tab."""
        if not self.has_professional_results():
            self.show_warning("Professional AutoML results not available. Run Professional AutoML first.")
            return

        # If dashboard instance exists, delegate to existing method
        if self.dashboard is not None:
            self.dashboard.render_professional_automl_tab()
            return

        # Otherwise show simplified version
        self._render_simplified()

    def _render_simplified(self) -> None:
        """Render simplified professional AutoML results."""
        st.subheader("🔥 Professional AutoML Results")

        professional_results = self.get_session_data("professional_results", {})

        if not professional_results:
            self.show_info("No professional results available.")
            return

        # Display results
        st.markdown("### 🎯 Optimization Results")

        # Best model info
        if "best_model" in professional_results:
            best_model = professional_results["best_model"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Best Model", best_model.get("model_name", "Unknown"))
            with col2:
                st.metric("📊 Score", f"{best_model.get('score', 0):.4f}")
            with col3:
                st.metric("⏱️ Training Time", f"{best_model.get('training_time', 0):.2f}s")

        # All results
        with st.expander("📋 **All Trial Results**", expanded=False):
            st.json(professional_results)
