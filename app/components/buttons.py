"""
Button components with consistent styling.
"""

from typing import Callable, Optional

import streamlit as st


class PrimaryButton:
    """Primary action button."""

    @staticmethod
    def render(
        label: str,
        on_click: Optional[Callable] = None,
        disabled: bool = False,
        width: str = "stretch",
        help_text: Optional[str] = None,
    ) -> bool:
        """
        Render a primary button.

        Args:
            label: Button label
            on_click: Callback function
            disabled: Whether button is disabled
            width: Button width ('stretch' or None)
            help_text: Optional help tooltip

        Returns:
            True if button was clicked
        """
        kwargs = {
            "type": "primary",
            "disabled": disabled,
        }

        if width == "stretch":
            kwargs["width"] = "stretch"

        if help_text:
            kwargs["help"] = help_text

        if on_click:
            kwargs["on_click"] = on_click

        return st.button(label, **kwargs)


class SecondaryButton:
    """Secondary action button."""

    @staticmethod
    def render(
        label: str,
        on_click: Optional[Callable] = None,
        disabled: bool = False,
        width: str = "stretch",
        help_text: Optional[str] = None,
    ) -> bool:
        """
        Render a secondary button.

        Args:
            label: Button label
            on_click: Callback function
            disabled: Whether button is disabled
            width: Button width ('stretch' or None)
            help_text: Optional help tooltip

        Returns:
            True if button was clicked
        """
        kwargs = {
            "disabled": disabled,
        }

        if width == "stretch":
            kwargs["width"] = "stretch"

        if help_text:
            kwargs["help"] = help_text

        if on_click:
            kwargs["on_click"] = on_click

        return st.button(label, **kwargs)
