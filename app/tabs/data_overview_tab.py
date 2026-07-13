"""
Data Overview Tab - Display dataset statistics and visualizations.
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.visualize import Visualizer

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class DataOverviewTab(BaseTab):
    """Tab for displaying dataset overview and statistics."""

    def __init__(self, dashboard_instance=None):
        """
        Initialize Data Overview tab.

        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance

    def render(self) -> None:
        """Render the data overview tab."""
        if not self.require_data():
            return

        data = self.get_session_data("data")

        # If dashboard instance exists, delegate to existing method for now
        if self.dashboard is not None:
            self.dashboard.render_data_overview()
            return

        # Otherwise use simplified version
        self._render_simplified_overview(data)

    def _render_simplified_overview(self, data: pd.DataFrame) -> None:
        """
        Render simplified data overview (fallback).

        Args:
            data: Dataset to display
        """
        st.subheader("📊 Dataset Overview")

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📝 Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("📊 Columns", f"{data.shape[1]}")
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
            st.metric("🔢 Numeric", numeric_cols)
        with col4:
            categorical_cols = data.select_dtypes(exclude=[np.number]).shape[1]
            st.metric("📝 Categorical", categorical_cols)

        st.markdown("---")

        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(data.head(10), width="stretch")

        # Data info
        with st.expander("ℹ️ **Column Information**"):
            st.markdown("#### Column Details")

            col_info = []
            for col in data.columns:
                col_info.append(
                    {
                        "Column": col,
                        "Type": str(data[col].dtype),
                        "Non-Null": data[col].count(),
                        "Null": data[col].isnull().sum(),
                        "Unique": data[col].nunique(),
                    }
                )

            info_df = pd.DataFrame(col_info)
            st.dataframe(info_df, width="stretch")

        # Visualizations
        st.markdown("### 📈 Data Visualizations")

        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            try:
                visualizer = Visualizer()
                fig = visualizer.plot_correlation_heatmap(numeric_data.iloc[:, :20])
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                self.show_warning(f"Could not generate correlation heatmap: {e}")
