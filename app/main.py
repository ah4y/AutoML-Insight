"""Main Streamlit application for AutoML-Insight."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ui_dashboard import AutoMLDashboard

def main():
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="AutoML-Insight",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<div class="main-header">🤖 AutoML-Insight</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.2rem; color: #666;'>
        Professional AutoML Platform for Automated Model Selection, Training & Explainability
        </p>
    """, unsafe_allow_html=True)
    
    # Show connection status if in remote mode
    if 'execution_mode' in st.session_state and st.session_state.execution_mode in ['remote', 'colab']:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.get('jupyter_connected', False):
                st.success("✅ Connected to Jupyter server", icon="🔗")
            else:
                st.info("ℹ️ Select 'Remote Jupyter Server' in sidebar to connect", icon="💡")
    
    # Initialize dashboard
    dashboard = AutoMLDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
