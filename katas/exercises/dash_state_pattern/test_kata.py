"""Tests for Dash State pattern kata."""

import dash
import plotly.graph_objects as go


def test_creates_dash_app():
    """Test that function returns a Dash app."""
    from template import create_state_app

    app = create_state_app()
    assert isinstance(app, dash.Dash)


def test_callback_registered():
    """Test that callback is registered."""
    from template import create_state_app

    app = create_state_app()
    assert len(app.callback_map) > 0


def test_callback_uses_state():
    """Test that callback uses State (not just Input)."""
    from template import create_state_app

    app = create_state_app()

    callback_id = list(app.callback_map.keys())[0]
    callback_info = app.callback_map[callback_id]

    # Should have State components
    assert len(callback_info['state']) > 0


def test_callback_input_is_button():
    """Test that Input is the button (not dropdown/slider)."""
    from template import create_state_app

    app = create_state_app()

    callback_id = list(app.callback_map.keys())[0]
    callback_info = app.callback_map[callback_id]

    # Button should be Input
    inputs = callback_info['inputs']
    assert len(inputs) == 1
    assert 'update-btn' in str(inputs[0]) or 'button' in str(inputs[0]).lower()


def test_callback_executes():
    """Test that callback executes and returns a figure."""
    from template import create_state_app

    app = create_state_app()

    with app.server.test_request_context():
        callback_id = list(app.callback_map.keys())[0]
        callback_fn = app.callback_map[callback_id]['callback']

        # Simulate button click with State values
        result = callback_fn(1, 'A', 40)
        assert isinstance(result, go.Figure)
        assert len(result.data[0].x) == 40
