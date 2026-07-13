"""
Metric card components for displaying statistics.
"""

import streamlit as st
from typing import Optional, List, Tuple


class MetricCard:
    """Display a single metric card."""
    
    @staticmethod
    def render(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
        """
        Render a metric card.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta value
            help_text: Optional help tooltip
        """
        st.metric(label=label, value=value, delta=delta, help=help_text)


class MetricRow:
    """Display multiple metrics in a row."""
    
    @staticmethod
    def render(metrics: List[Tuple[str, str, Optional[str]]]):
        """
        Render a row of metrics.
        
        Args:
            metrics: List of (label, value, delta) tuples
        """
        cols = st.columns(len(metrics))
        
        for col, (label, value, delta) in zip(cols, metrics):
            with col:
                st.metric(label=label, value=value, delta=delta)
    
    @staticmethod
    def render_4_column(metric1: Tuple, metric2: Tuple, metric3: Tuple, metric4: Tuple):
        """
        Render 4 metrics in columns.
        
        Args:
            metric1-4: Tuples of (label, value, delta)
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(*metric1)
        with col2:
            st.metric(*metric2)
        with col3:
            st.metric(*metric3)
        with col4:
            st.metric(*metric4)
