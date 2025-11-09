"""Tests for Dash error handling kata."""

import dash
import plotly.graph_objects as go


def test_creates_dash_app():
    """Test that function returns a Dash app."""
    from template import create_validated_app

    app = create_validated_app()
    assert isinstance(app, dash.Dash)


def test_callback_registered():
    """Test that callback is registered."""
    from template import create_validated_app

    app = create_validated_app()
    assert len(app.callback_map) > 0


def test_callback_handles_none_input():
    """Test that callback returns error for None input."""
    from template import create_validated_app

    app = create_validated_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, msg = callback_fn(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # Empty figure
        assert 'error' in msg.lower() or 'Error' in msg


def test_callback_handles_invalid_range():
    """Test that callback returns error for n < 1."""
    from template import create_validated_app

    app = create_validated_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, msg = callback_fn(0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert 'error' in msg.lower() or 'Error' in msg


def test_callback_handles_too_many_points():
    """Test that callback returns error for n > 100."""
    from template import create_validated_app

    app = create_validated_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, msg = callback_fn(101)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert 'Too many' in msg or 'max 100' in msg


def test_callback_handles_valid_input():
    """Test that callback returns chart for valid input."""
    from template import create_validated_app

    app = create_validated_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        fig, msg = callback_fn(20)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have data
        assert 'Success' in msg or 'success' in msg.lower() or '20' in msg
