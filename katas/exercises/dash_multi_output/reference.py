"""Dash multi-output callback kata - reference solution."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_multi_output_app() -> dash.Dash:
    """Create app with callback that updates both graph and statistics text."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Multi-Output Demo'),
        dcc.Slider(
            id='n-points',
            min=5,
            max=50,
            value=20,
            step=5,
            marks={i: str(i) for i in range(5, 51, 5)}
        ),
        dcc.Graph(id='chart'),
        html.Div(id='stats-text')
    ])

    @app.callback(
        Output('chart', 'figure'),
        Output('stats-text', 'children'),
        Input('n-points', 'value')
    )
    def update_outputs(n_points):
        x = list(range(n_points))
        y = [i**2 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(title=f'Quadratic Function ({n_points} points)')

        total = sum(y)
        mean = total / len(y)
        stats_text = f'Points: {n_points}, Sum: {total}, Mean: {mean:.2f}'

        return fig, stats_text

    return app
