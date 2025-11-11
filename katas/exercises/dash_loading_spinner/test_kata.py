"""Tests for Dash loading spinner kata."""

import dash


try:
    from user_kata import create_app_with_loading
except ImportError:
    from .reference import create_app_with_loading


def test_creates_dash_app():
    """Test that function returns a Dash app."""

    app = create_app_with_loading()
    assert isinstance(app, dash.Dash)


def test_layout_exists():
    """Test that app has a layout."""

    app = create_app_with_loading()
    assert app.layout is not None


def test_layout_contains_loading_component():
    """Test that layout contains Loading component."""

    app = create_app_with_loading()
    layout_str = str(app.layout)
    assert 'Loading' in layout_str


def test_layout_contains_graph():
    """Test that layout contains graph with correct ID."""

    app = create_app_with_loading()
    layout_str = str(app.layout)
    assert 'output-chart' in layout_str


def test_layout_contains_dropdown():
    """Test that layout contains dropdown."""

    app = create_app_with_loading()
    layout_str = str(app.layout)
    assert 'dataset' in layout_str
