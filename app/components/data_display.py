"""
Data display components.
"""

from typing import Optional

import pandas as pd
import streamlit as st


class DataTable:
    """Display data in a table format."""

    @staticmethod
    def render(data: pd.DataFrame, max_rows: Optional[int] = None, use_container_width: bool = True):
        """
        Render a data table.

        Args:
            data: DataFrame to display
            max_rows: Maximum rows to show
            use_container_width: Use full container width
        """
        if max_rows:
            st.dataframe(data.head(max_rows), width="stretch" if use_container_width else None)
        else:
            st.dataframe(data, width="stretch" if use_container_width else None)


class DataPreview:
    """Display a preview of the dataset."""

    @staticmethod
    def render(data: pd.DataFrame, n_rows: int = 10, show_info: bool = True):
        """
        Render a data preview.

        Args:
            data: DataFrame to preview
            n_rows: Number of rows to show
            show_info: Show additional info
        """
        if show_info:
            st.markdown(f"**Showing first {n_rows} rows of {data.shape[0]:,} total rows**")

        st.dataframe(data.head(n_rows), width="stretch")

        if show_info:
            st.caption(f"Total: {data.shape[0]:,} rows × {data.shape[1]} columns")
