"""Dash multi-output callback kata - updating multiple components."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_multi_output_app() -> dash.Dash:
    """Create app with callback that updates both graph and statistics text.

    Layout should have:
    - Slider id='n-points' with min=5, max=50, value=20, step=5
    - Graph id='chart'
    - Div id='stats-text' for displaying statistics

    Callback should:
    - Input: 'n-points' value
    - Output 1: 'chart' figure (line plot of y=x^2)
    - Output 2: 'stats-text' children (text showing: "Points: {n}, Sum: {sum}, Mean: {mean:.2f}")
    - Return tuple: (figure, stats_text)

    Returns:
        Dash app with multi-output callback
    """
    # BLANK_START
    raise NotImplementedError(
        "Use multiple Output() in callback, return tuple of (figure, text)"
    )
    # BLANK_END
