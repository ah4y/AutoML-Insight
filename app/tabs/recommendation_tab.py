"""
Recommendation Tab - Display meta-learning recommendations.
"""

import streamlit as st
from typing import Optional
import logging

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class RecommendationTab(BaseTab):
    """Tab for displaying model recommendations."""
    
    def __init__(self, dashboard_instance=None):
        """
        Initialize Recommendation tab.
        
        Args:
            dashboard_instance: Reference to main dashboard for backward compatibility
        """
        super().__init__()
        self.dashboard = dashboard_instance
    
    def render(self) -> None:
        """Render the recommendation tab."""
        # If dashboard instance exists, delegate to existing method
        if self.dashboard is not None:
            self.dashboard.render_recommendation()
            return
        
        # Otherwise show simplified version
        self._render_simplified()
    
    def _render_simplified(self) -> None:
        """Render simplified recommendations."""
        st.subheader("🎯 Model Recommendations")
        
        if not self.require_results():
            return
        
        # Check for AI insights
        ai_insights = self.get_session_data('ai_insights')
        
        if ai_insights and 'recommended_models' in ai_insights:
            st.markdown("### 🤖 AI-Powered Recommendations")
            
            models = ai_insights['recommended_models']
            if isinstance(models, list):
                for i, model in enumerate(models, 1):
                    st.success(f"{i}. {model}")
            else:
                st.success(models)
        else:
            st.markdown("### 📊 Meta-Learning Recommendations")
            self.show_info("Recommendations based on dataset characteristics.")
            
            # Show dataset profile
            profile = self.get_session_data('profile', {})
            
            if profile:
                with st.expander("📋 **Dataset Profile**"):
                    st.json(profile)
