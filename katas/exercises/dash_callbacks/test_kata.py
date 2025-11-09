"""Tests for Dash callbacks kata."""

import pytest
import dash
from dash import dcc, html
import plotly.graph_objects as go


def test_create_basic_app():
    """Test basic app creation with dropdown and graph."""
    from template import create_basic_app

    app = create_basic_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    # Verify layout contains required components
    layout_str = str(app.layout)
    assert 'basic-chart' in layout_str
    assert 'chart-type' in layout_str


def test_add_single_input_callback():
    """Test single input callback updates graph correctly."""
    from template import add_single_input_callback, create_basic_app

    app = create_basic_app()
    add_single_input_callback(app)

    # Verify callback is registered
    assert len(app.callback_map) > 0

    # Test callback execution
    with app.server.test_request_context():
        # Get callback function
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test with 'line' chart type
        result = callback_fn('line')
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

        # Test with 'bar' chart type
        result = callback_fn('bar')
        assert isinstance(result, go.Figure)
        assert result.data[0].type == 'bar'


def test_create_multi_input_app():
    """Test multi-input app creation."""
    from template import create_multi_input_app

    app = create_multi_input_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    layout_str = str(app.layout)
    assert 'multi-chart' in layout_str
    assert 'dataset' in layout_str
    assert 'n-points' in layout_str


def test_add_multi_input_callback():
    """Test callback with multiple inputs."""
    from template import add_multi_input_callback, create_multi_input_app

    app = create_multi_input_app()
    add_multi_input_callback(app)

    assert len(app.callback_map) > 0

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test dataset A with 30 points
        result = callback_fn('A', 30)
        assert isinstance(result, go.Figure)
        assert len(result.data[0].x) == 30
        # Verify it's quadratic (y = x^2)
        assert result.data[0].y[5] == 25  # 5^2 = 25

        # Test dataset B with 20 points
        result = callback_fn('B', 20)
        assert len(result.data[0].x) == 20
        # Verify it's cubic (y = x^3)
        assert result.data[0].y[3] == 27  # 3^3 = 27


def test_create_state_app():
    """Test app creation with State components."""
    from template import create_state_app

    app = create_state_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    layout_str = str(app.layout)
    assert 'state-chart' in layout_str
    assert 'state-dataset' in layout_str
    assert 'state-n-points' in layout_str
    assert 'update-button' in layout_str


def test_add_state_callback():
    """Test State callback only triggers on button click."""
    from template import add_state_callback, create_state_app

    app = create_state_app()
    add_state_callback(app)

    assert len(app.callback_map) > 0

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test callback with different click counts and states
        result = callback_fn(0, 'A', 40)
        assert isinstance(result, go.Figure)
        assert len(result.data[0].x) == 40

        result = callback_fn(1, 'B', 30)
        assert len(result.data[0].x) == 30


def test_create_multi_output_app():
    """Test app creation with multiple outputs."""
    from template import create_multi_output_app

    app = create_multi_output_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    layout_str = str(app.layout)
    assert 'output-chart' in layout_str
    assert 'output-stats' in layout_str
    assert 'output-n-points' in layout_str


def test_add_multi_output_callback():
    """Test callback with multiple outputs."""
    from template import add_multi_output_callback, create_multi_output_app

    app = create_multi_output_app()
    add_multi_output_callback(app)

    assert len(app.callback_map) > 0

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test with 10 points
        fig, stats = callback_fn(10)
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, str)
        assert 'Points: 10' in stats
        assert 'Sum:' in stats
        assert 'Mean:' in stats

        # Verify statistics are correct
        # For 10 points (0-9), sum of squares = 0+1+4+9+16+25+36+49+64+81 = 285
        assert '285' in stats


def test_create_loading_app():
    """Test app creation with loading component."""
    from template import create_loading_app

    app = create_loading_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    layout_str = str(app.layout)
    assert 'loading-chart' in layout_str
    assert 'loading-dataset' in layout_str
    assert 'Loading' in layout_str


def test_add_loading_callback():
    """Test callback with loading state."""
    from template import add_loading_callback, create_loading_app

    app = create_loading_app()
    add_loading_callback(app)

    assert len(app.callback_map) > 0

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test each dataset type
        result_a = callback_fn('A')
        assert isinstance(result_a, go.Figure)
        assert len(result_a.data) > 0

        result_b = callback_fn('B')
        assert isinstance(result_b, go.Figure)

        result_c = callback_fn('C')
        assert isinstance(result_c, go.Figure)
        assert result_c.data[0].type == 'bar'


def test_create_error_handling_app():
    """Test app creation with error handling."""
    from template import create_error_handling_app

    app = create_error_handling_app()

    assert isinstance(app, dash.Dash)
    assert app.layout is not None

    layout_str = str(app.layout)
    assert 'error-chart' in layout_str
    assert 'error-message' in layout_str
    assert 'error-input' in layout_str


def test_add_error_handling_callback():
    """Test callback with proper error handling."""
    from template import add_error_handling_callback, create_error_handling_app

    app = create_error_handling_app()
    add_error_handling_callback(app)

    assert len(app.callback_map) > 0

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test None input
        fig, msg = callback_fn(None)
        assert isinstance(fig, go.Figure)
        assert 'Error' in msg or 'error' in msg.lower()

        # Test invalid input (< 1)
        fig, msg = callback_fn(0)
        assert isinstance(fig, go.Figure)
        assert 'Error' in msg or 'error' in msg.lower()

        # Test too many points
        fig, msg = callback_fn(101)
        assert isinstance(fig, go.Figure)
        assert 'Too many points' in msg or 'max 100' in msg

        # Test valid input
        fig, msg = callback_fn(20)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Success' in msg or 'success' in msg.lower() or '20' in msg


def test_callback_outputs_match_inputs():
    """Test that callback signatures match decorator definitions."""
    from template import (
        add_single_input_callback,
        create_basic_app,
        add_multi_input_callback,
        create_multi_input_app,
    )

    # Test basic app
    app1 = create_basic_app()
    add_single_input_callback(app1)

    # Test multi-input app
    app2 = create_multi_input_app()
    add_multi_input_callback(app2)

    # Verify callbacks are properly registered
    assert len(app1.callback_map) == 1
    assert len(app2.callback_map) == 1


def test_state_vs_input_behavior():
    """Test that State doesn't trigger callback, only Input does."""
    from template import add_state_callback, create_state_app

    app = create_state_app()
    add_state_callback(app)

    # Get callback inputs/states
    callback_id = list(app.callback_map.keys())[0]
    callback_info = app.callback_map[callback_id]

    # Verify Input is on button, State is on dropdown/slider
    inputs = callback_info['inputs']
    states = callback_info['state']

    # Should have 1 Input (button) and 2 States (dropdown, slider)
    assert len(inputs) == 1
    assert len(states) == 2

    # Button should be the Input
    assert 'update-button' in str(inputs[0])
