"""Dash State pattern kata - reference solution."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go


def create_state_app() -> dash.Dash:
    """Create app where graph only updates when button is clicked."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('State Pattern Demo'),
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
            step=10
        ),
        html.Button('Update Chart', id='update-btn', n_clicks=0),
        dcc.Graph(id='result-chart')
    ])

    @app.callback(
        Output('result-chart', 'figure'),
        Input('update-btn', 'n_clicks'),
        State('dataset', 'value'),
        State('n-points', 'value')
    )
    def update_chart(n_clicks, dataset, n_points):
        x = list(range(n_points))

        if dataset == 'A':
            y = [i**2 for i in x]
        else:
            y = [i**3 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(title=f'Dataset {dataset} - Clicks: {n_clicks}')
        return fig

    return app
