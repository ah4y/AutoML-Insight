"""
Section header components.
"""

import streamlit as st
from typing import Optional


class SectionHeader:
    """Render a main section header."""
    
    @staticmethod
    def render(title: str, description: Optional[str] = None, icon: Optional[str] = None):
        """
        Render a section header.
        
        Args:
            title: Section title
            description: Optional description
            icon: Optional icon emoji
        """
        if icon:
            st.markdown(f"### {icon} {title}")
        else:
            st.markdown(f"### {title}")
        
        if description:
            st.markdown(description)
        
        st.markdown("---")


class SubsectionHeader:
    """Render a subsection header."""
    
    @staticmethod
    def render(title: str, description: Optional[str] = None):
        """
        Render a subsection header.
        
        Args:
            title: Subsection title
            description: Optional description
        """
        st.markdown(f"#### {title}")
        
        if description:
            st.markdown(description)
