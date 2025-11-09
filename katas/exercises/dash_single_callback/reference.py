"""Dash single callback kata - reference solution."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def create_app_with_callback() -> dash.Dash:
    """Create app and add callback that updates graph based on dropdown."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Interactive Chart'),
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

    @app.callback(
        Output('main-chart', 'figure'),
        Input('chart-type', 'value')
    )
    def update_chart(chart_type):
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]

        if chart_type == 'line':
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        else:
            fig = go.Figure(data=go.Bar(x=x, y=y))

        fig.update_layout(title=f'{chart_type.title()} Chart')
        return fig

    return app
