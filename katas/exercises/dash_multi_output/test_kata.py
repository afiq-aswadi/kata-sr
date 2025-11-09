"""Tests for Dash multi-output callback kata."""

import dash
import plotly.graph_objects as go


def test_creates_dash_app():
    """Test that function returns a Dash app."""
    from template import create_multi_output_app

    app = create_multi_output_app()
    assert isinstance(app, dash.Dash)


def test_callback_registered():
    """Test that callback is registered."""
    from template import create_multi_output_app

    app = create_multi_output_app()
    assert len(app.callback_map) > 0


def test_callback_returns_tuple():
    """Test that callback returns two values (figure and text)."""
    from template import create_multi_output_app

    app = create_multi_output_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn(10)
        assert isinstance(result, tuple)
        assert len(result) == 2


def test_callback_first_output_is_figure():
    """Test that first output is a Plotly figure."""
    from template import create_multi_output_app

    app = create_multi_output_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, text = callback_fn(10)
        assert isinstance(fig, go.Figure)


def test_callback_second_output_is_stats():
    """Test that second output contains statistics."""
    from template import create_multi_output_app

    app = create_multi_output_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, text = callback_fn(10)
        assert isinstance(text, str)
        assert 'Points: 10' in text
        assert 'Sum:' in text
        assert 'Mean:' in text
        # For 10 points (0-9), sum = 0+1+4+9+16+25+36+49+64+81 = 285
        assert '285' in text
