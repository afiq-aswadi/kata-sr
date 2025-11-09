"""Dash multi-input callback kata - reference solution."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_multi_input_app() -> dash.Dash:
    """Create app with callback that takes dropdown and slider inputs."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Multi-Input Dashboard'),
        dcc.Dropdown(
            id='dataset',
            options=[
                {'label': 'Dataset A', 'value': 'A'},
                {'label': 'Dataset B', 'value': 'B'}
            ],
            value='A'
        ),
        dcc.Slider(
            id='n-points',
            min=10,
            max=100,
            value=50,
            step=10,
            marks={i: str(i) for i in range(10, 101, 10)}
        ),
        dcc.Graph(id='output-chart')
    ])

    @app.callback(
        Output('output-chart', 'figure'),
        Input('dataset', 'value'),
        Input('n-points', 'value')
    )
    def update_chart(dataset, n_points):
        x = list(range(n_points))

        if dataset == 'A':
            y = [i**2 for i in x]
        else:
            y = [i**3 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(title=f'Dataset {dataset} ({n_points} points)')
        return fig

    return app
