"""Dash basic layout kata - creating app structure with components."""

import dash
from dash import dcc, html


def create_dashboard_layout() -> dash.Dash:
    """Create a basic Dash app with title, dropdown, and graph.

    Build a layout containing:
    - An H1 title saying "My Dashboard"
    - A Dropdown with id='chart-type', options for 'line' and 'bar', default value 'line'
    - A Graph with id='main-chart'

    Returns:
        Dash app instance with configured layout
    """
    # BLANK_START
    raise NotImplementedError(
        "Create dash.Dash(__name__), set app.layout to html.Div with components"
    )
    # BLANK_END
