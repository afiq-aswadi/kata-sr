"""Dash State pattern kata - button-triggered updates."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go


def create_state_app() -> dash.Dash:
    """Create app where graph only updates when button is clicked.

    Layout should have:
    - Dropdown id='dataset' with options 'A' and 'B'
    - Slider id='n-points' with min=10, max=100, value=50
    - Button id='update-btn' with text 'Update Chart'
    - Graph id='result-chart'

    Callback should:
    - Input: 'update-btn' n_clicks property (triggers the callback)
    - State: 'dataset' value (doesn't trigger, just provides value)
    - State: 'n-points' value (doesn't trigger, just provides value)
    - Generate plot only when button is clicked, not when dropdown/slider change

    Returns:
        Dash app with State-based callback
    """
    # BLANK_START
    raise NotImplementedError(
        "Use Input() for button, State() for dropdown and slider"
    )
    # BLANK_END
