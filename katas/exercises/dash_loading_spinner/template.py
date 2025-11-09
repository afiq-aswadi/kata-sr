"""Dash loading spinner kata - adding loading states."""

import dash
from dash import dcc, html


def create_app_with_loading() -> dash.Dash:
    """Create app with a loading spinner wrapping the graph.

    Layout should have:
    - H1 title "Loading Demo"
    - Dropdown id='dataset' with options 'A', 'B', 'C'
    - dcc.Loading component wrapping dcc.Graph id='output-chart'

    The Loading component should contain the Graph as its children.
    This shows a spinner automatically when the graph is updating.

    Returns:
        Dash app with Loading component in layout
    """
    # BLANK_START
    raise NotImplementedError(
        "Use dcc.Loading(children=[dcc.Graph(...)]) to wrap the graph"
    )
    # BLANK_END
