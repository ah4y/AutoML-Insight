"""
PCA Analysis Tab - Display dimensionality reduction results.
"""

import logging

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class PCAAnalysisTab(BaseTab):
    """Tab for displaying PCA analysis results."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize PCA Analysis tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the PCA analysis tab."""
        # If dashboard instance exists, delegate to existing method
        if self.dashboard is not None:
            self.dashboard.render_pca_analysis()
            return

        # Otherwise show simplified version
        self._render_simplified()

    def _render_simplified(self) -> None:
        """Render simplified PCA analysis."""
        st.subheader("📐 PCA Analysis")

        if not self.has_results() and not self.has_professional_results():
            self.show_info("Run AutoML first to see PCA analysis.")
            return

        dimred_results = self.get_session_data("dimred_results")

        if dimred_results:
            st.markdown("### 📊 Dimensionality Reduction Results")

            # Display reduction info
            if "method" in dimred_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Method", dimred_results.get("method", "PCA"))
                with col2:
                    st.metric("Components", dimred_results.get("n_components", "N/A"))
                with col3:
                    variance = dimred_results.get("explained_variance", 0)
                    st.metric("Variance Explained", f"{variance:.2%}")

            with st.expander("📋 **Full Results**"):
                st.json(dimred_results)
        else:
            self.show_info("No dimensionality reduction results available.")
