"""Tests for Dash basic layout kata."""

import dash


try:
    from user_kata import create_dashboard_layout
except ImportError:
    from .reference import create_dashboard_layout


def test_creates_dash_app():
    """Test that function returns a Dash app instance."""

    app = create_dashboard_layout()
    assert isinstance(app, dash.Dash)


def test_layout_exists():
    """Test that app has a layout configured."""

    app = create_dashboard_layout()
    assert app.layout is not None


def test_layout_contains_title():
    """Test that layout contains H1 title."""

    app = create_dashboard_layout()
    layout_str = str(app.layout)
    assert 'My Dashboard' in layout_str or 'H1' in layout_str


def test_layout_contains_dropdown():
    """Test that layout contains dropdown with correct ID."""

    app = create_dashboard_layout()
    layout_str = str(app.layout)
    assert 'chart-type' in layout_str
    assert 'Dropdown' in layout_str


def test_layout_contains_graph():
    """Test that layout contains graph with correct ID."""

    app = create_dashboard_layout()
    layout_str = str(app.layout)
    assert 'main-chart' in layout_str
    assert 'Graph' in layout_str
