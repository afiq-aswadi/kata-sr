"""Tests for Dash multi-input callback kata."""

import dash
import plotly.graph_objects as go


try:
    from user_kata import create_multi_input_app
except ImportError:
    from .reference import create_multi_input_app


def test_creates_dash_app():
    """Test that function returns a Dash app."""

    app = create_multi_input_app()
    assert isinstance(app, dash.Dash)


def test_callback_registered():
    """Test that callback is registered."""

    app = create_multi_input_app()
    assert len(app.callback_map) > 0


def test_callback_accepts_two_inputs():
    """Test that callback function accepts two arguments."""

    app = create_multi_input_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Test with dataset A and 30 points
        result = callback_fn('A', 30)
        assert isinstance(result, go.Figure)


def test_callback_dataset_a():
    """Test callback generates correct data for dataset A."""

    app = create_multi_input_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn('A', 30)
        assert len(result.data[0].x) == 30
        # Verify quadratic: y = x^2
        assert result.data[0].y[5] == 25  # 5^2 = 25


def test_callback_dataset_b():
    """Test callback generates correct data for dataset B."""

    app = create_multi_input_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn('B', 20)
        assert len(result.data[0].x) == 20
        # Verify cubic: y = x^3
        assert result.data[0].y[3] == 27  # 3^3 = 27
