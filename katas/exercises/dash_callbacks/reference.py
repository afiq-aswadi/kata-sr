"""Dash callbacks kata - reference solution."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go


def create_basic_app() -> dash.Dash:
    """Create a basic Dash app with a dropdown and graph."""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('Basic Interactive Dashboard'),
        dcc.Dropdown(
            id='chart-type',
            options=[
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Bar Chart', 'value': 'bar'}
            ],
            value='line'
        ),
        dcc.Graph(id='basic-chart')
    ])
    return app


def add_single_input_callback(app: dash.Dash) -> None:
    """Add callback that updates graph based on dropdown selection."""
    @app.callback(
        Output('basic-chart', 'figure'),
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


def create_multi_input_app() -> dash.Dash:
    """Create Dash app with multiple inputs (dropdown + slider)."""
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
        dcc.Graph(id='multi-chart')
    ])
    return app


def add_multi_input_callback(app: dash.Dash) -> None:
    """Add callback with multiple inputs (dropdown + slider)."""
    @app.callback(
        Output('multi-chart', 'figure'),
        Input('dataset', 'value'),
        Input('n-points', 'value')
    )
    def update_multi_chart(dataset, n_points):
        x = list(range(n_points))

        if dataset == 'A':
            y = [i**2 for i in x]
        else:
            y = [i**3 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(title=f'Dataset {dataset} ({n_points} points)')
        return fig


def create_state_app() -> dash.Dash:
    """Create Dash app using State for button-triggered updates."""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('State vs Input Demo'),
        dcc.Dropdown(
            id='state-dataset',
            options=[
                {'label': 'Dataset A', 'value': 'A'},
                {'label': 'Dataset B', 'value': 'B'}
            ],
            value='A'
        ),
        dcc.Slider(
            id='state-n-points',
            min=10,
            max=100,
            value=50,
            step=10,
            marks={i: str(i) for i in range(10, 101, 10)}
        ),
        html.Button('Update Chart', id='update-button', n_clicks=0),
        dcc.Graph(id='state-chart')
    ])
    return app


def add_state_callback(app: dash.Dash) -> None:
    """Add callback using State (doesn't trigger on change, only on button click)."""
    @app.callback(
        Output('state-chart', 'figure'),
        Input('update-button', 'n_clicks'),
        State('state-dataset', 'value'),
        State('state-n-points', 'value')
    )
    def update_state_chart(n_clicks, dataset, n_points):
        x = list(range(n_points))

        if dataset == 'A':
            y = [i**2 for i in x]
        else:
            y = [i**3 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(title=f'Dataset {dataset} ({n_points} points) - Clicks: {n_clicks}')
        return fig


def create_multi_output_app() -> dash.Dash:
    """Create Dash app with callback that updates multiple outputs."""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('Multiple Outputs Demo'),
        dcc.Slider(
            id='output-n-points',
            min=5,
            max=50,
            value=20,
            step=5,
            marks={i: str(i) for i in range(5, 51, 5)}
        ),
        dcc.Graph(id='output-chart'),
        html.Div(id='output-stats')
    ])
    return app


def add_multi_output_callback(app: dash.Dash) -> None:
    """Add callback that updates multiple outputs (graph + text)."""
    @app.callback(
        Output('output-chart', 'figure'),
        Output('output-stats', 'children'),
        Input('output-n-points', 'value')
    )
    def update_multi_output(n_points):
        x = list(range(n_points))
        y = [i**2 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(title=f'Quadratic Function ({n_points} points)')

        total = sum(y)
        mean = total / len(y)
        stats_text = f'Points: {n_points}, Sum: {total}, Mean: {mean:.2f}'

        return fig, stats_text


def create_loading_app() -> dash.Dash:
    """Create Dash app with loading spinner."""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('Loading State Demo'),
        dcc.Dropdown(
            id='loading-dataset',
            options=[
                {'label': 'Dataset A', 'value': 'A'},
                {'label': 'Dataset B', 'value': 'B'},
                {'label': 'Dataset C', 'value': 'C'}
            ],
            value='A'
        ),
        dcc.Loading(
            children=[dcc.Graph(id='loading-chart')]
        )
    ])
    return app


def add_loading_callback(app: dash.Dash) -> None:
    """Add callback that simulates slow computation with loading state."""
    @app.callback(
        Output('loading-chart', 'figure'),
        Input('loading-dataset', 'value')
    )
    def update_loading_chart(dataset):
        n = 30
        x = list(range(n))

        if dataset == 'A':
            y = [i**2 for i in x]
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        elif dataset == 'B':
            y = [2*i + 3 for i in x]
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))
        else:  # C
            y = [i**3 for i in x]
            fig = go.Figure(data=go.Bar(x=x, y=y))

        fig.update_layout(title=f'Dataset {dataset}')
        return fig


def create_error_handling_app() -> dash.Dash:
    """Create Dash app with error handling in callback."""
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('Error Handling Demo'),
        dcc.Input(
            id='error-input',
            type='number',
            value=10,
            placeholder='Enter number of points'
        ),
        dcc.Graph(id='error-chart'),
        html.Div(id='error-message')
    ])
    return app


def add_error_handling_callback(app: dash.Dash) -> None:
    """Add callback with error handling for invalid inputs."""
    @app.callback(
        Output('error-chart', 'figure'),
        Output('error-message', 'children'),
        Input('error-input', 'value')
    )
    def update_error_chart(n):
        # Handle edge cases
        if n is None or n < 1:
            return go.Figure(), 'Error: Please enter a valid number (>= 1)'

        if n > 100:
            return go.Figure(), 'Error: Too many points (max 100)'

        # Valid input
        x = list(range(n))
        y = [i**2 for i in x]

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
        fig.update_layout(title=f'Quadratic Function ({n} points)')

        return fig, f'Success: Displaying {n} points'
