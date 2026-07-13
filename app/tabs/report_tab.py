"""
Report Tab - Generate and download PDF reports.
"""

import logging

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ReportTab(BaseTab):
    """Tab for generating and downloading reports."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize Report tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the report tab."""
        # If dashboard instance exists, delegate to existing method
        if self.dashboard is not None:
            self.dashboard.render_report()
            return

        # Otherwise show simplified version
        self._render_simplified()

    def _render_simplified(self) -> None:
        """Render simplified report generation."""
        st.subheader("📄 Report Generation")

        if not self.require_results():
            return

        st.markdown("### 📊 Generate PDF Report")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.info("Generate a comprehensive PDF report of your AutoML results.")

            # Report options
            include_visualizations = st.checkbox("Include Visualizations", value=True)
            include_explainability = st.checkbox("Include SHAP Analysis", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)

        with col2:
            st.markdown("#### Report Preview")
            st.info("📄 Executive Summary\n📊 Model Results\n📈 Visualizations\n🎯 Recommendations")

        if st.button("🎯 **Generate Report**", type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Placeholder for report generation
                    self.show_success("Report generated successfully!")

                    # Download button placeholder
                    st.download_button(
                        label="📥 Download Report",
                        data=b"PDF content would go here",
                        file_name="automl_report.pdf",
                        mime="application/pdf",
                    )

                except Exception as e:
                    self.show_error(f"Failed to generate report: {e}")
