"""Dash error handling kata - reference solution."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_validated_app() -> dash.Dash:
    """Create app with input validation and error messages."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Error Handling Demo'),
        dcc.Input(
            id='n-input',
            type='number',
            value=10,
            placeholder='Enter number of points'
        ),
        dcc.Graph(id='result-chart'),
        html.Div(id='error-msg')
    ])

    @app.callback(
        Output('result-chart', 'figure'),
        Output('error-msg', 'children'),
        Input('n-input', 'value')
    )
    def update_chart(n):
        # Validate input
        if n is None or n < 1:
            return go.Figure(), 'Error: Please enter a valid number (>= 1)'

        if n > 100:
            return go.Figure(), 'Error: Too many points (max 100)'

        # Valid input - create chart
        x = list(range(n))
        y = [i**2 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(title=f'Quadratic Function ({n} points)')

        return fig, f'Success: Displaying {n} points'

    return app
