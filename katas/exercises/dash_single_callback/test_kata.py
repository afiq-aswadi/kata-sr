"""Tests for Dash single callback kata."""

import dash
import plotly.graph_objects as go


def test_creates_dash_app():
    """Test that function returns a Dash app."""
    from template import create_app_with_callback

    app = create_app_with_callback()
    assert isinstance(app, dash.Dash)


def test_callback_registered():
    """Test that at least one callback is registered."""
    from template import create_app_with_callback

    app = create_app_with_callback()
    assert len(app.callback_map) > 0


def test_callback_returns_figure():
    """Test that callback returns a Plotly figure."""
    from template import create_app_with_callback

    app = create_app_with_callback()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn('line')
        assert isinstance(result, go.Figure)


def test_callback_handles_line_chart():
    """Test callback creates line chart correctly."""
    from template import create_app_with_callback

    app = create_app_with_callback()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn('line')
        assert len(result.data) > 0
        assert result.data[0].type in ['scatter', None]  # scatter is default


def test_callback_handles_bar_chart():
    """Test callback creates bar chart correctly."""
    from template import create_app_with_callback

    app = create_app_with_callback()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        result = callback_fn('bar')
        assert len(result.data) > 0
        assert result.data[0].type == 'bar'
