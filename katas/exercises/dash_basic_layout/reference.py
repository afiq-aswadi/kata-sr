"""Dash basic layout kata - reference solution."""

import dash
from dash import dcc, html


def create_dashboard_layout() -> dash.Dash:
    """Create a basic Dash app with title, dropdown, and graph."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('My Dashboard'),
        dcc.Dropdown(
            id='chart-type',
            options=[
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Bar Chart', 'value': 'bar'}
            ],
            value='line'
        ),
        dcc.Graph(id='main-chart')
    ])

    return app
