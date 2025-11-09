"""Dash loading spinner kata - reference solution."""

import dash
from dash import dcc, html


def create_app_with_loading() -> dash.Dash:
    """Create app with a loading spinner wrapping the graph."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Loading Demo'),
        dcc.Dropdown(
            id='dataset',
            options=[
                {'label': 'Dataset A', 'value': 'A'},
                {'label': 'Dataset B', 'value': 'B'},
                {'label': 'Dataset C', 'value': 'C'}
            ],
            value='A'
        ),
        dcc.Loading(
            children=[dcc.Graph(id='output-chart')]
        )
    ])

    return app
