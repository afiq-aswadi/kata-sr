"""Dash multi-input callback kata - handling multiple inputs."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_multi_input_app() -> dash.Dash:
    """Create app with callback that takes dropdown and slider inputs.

    Layout should have:
    - Dropdown id='dataset' with options 'A' and 'B', default 'A'
    - Slider id='n-points' with min=10, max=100, value=50, step=10
    - Graph id='output-chart'

    Callback should:
    - Take both inputs (dataset, n_points)
    - Generate x = range(n_points)
    - For dataset A: y = x^2, for dataset B: y = x^3
    - Return line plot with title showing dataset and n_points

    Returns:
        Dash app with layout and callback configured
    """
    # BLANK_START
    raise NotImplementedError(
        "Create app with layout, add callback with multiple Input() in a list"
    )
    # BLANK_END
