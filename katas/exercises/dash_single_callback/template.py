"""Dash single callback kata - connecting inputs to outputs."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_app_with_callback() -> dash.Dash:
    """Create app and add callback that updates graph based on dropdown.

    The callback should:
    - Take input from 'chart-type' dropdown's 'value' property
    - Update 'main-chart' graph's 'figure' property
    - Return a line chart if chart_type=='line', bar chart if chart_type=='bar'
    - Use data: x=[1,2,3,4,5], y=[1,4,9,16,25]

    Returns:
        Dash app with layout and callback configured
    """
    # BLANK_START
    raise NotImplementedError(
        "Create app with layout, then add @app.callback decorator with Input/Output"
    )
    # BLANK_END
