"""Tests for plotly_scatter kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_scatter_plot
except ImportError:
    from .reference import create_scatter_plot


def test_returns_figure():
    """Should return a go.Figure object"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_scatter_plot(x, y, "Test")
    assert isinstance(fig, go.Figure)


def test_has_scatter_trace():
    """Figure should contain a scatter trace"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_scatter_plot(x, y, "Test")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_scatter_mode_is_markers():
    """Scatter trace should use markers mode"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_scatter_plot(x, y, "Test")
    assert fig.data[0].mode == "markers"


def test_correct_data():
    """Trace should contain the correct x and y data"""
    x = [1, 2, 3, 4]
    y = [10, 20, 15, 25]
    fig = create_scatter_plot(x, y, "Test")
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_title_set():
    """Figure should have the specified title"""
    x = [1, 2]
    y = [3, 4]
    title = "My Scatter Plot"
    fig = create_scatter_plot(x, y, title)
    assert fig.layout.title.text == title


def test_default_marker_styling():
    """Should use default marker color and size"""
    x = [1, 2]
    y = [3, 4]
    fig = create_scatter_plot(x, y, "Test")
    assert fig.data[0].marker.color == "blue"
    assert fig.data[0].marker.size == 10


def test_custom_marker_color():
    """Should accept custom marker color"""
    x = [1, 2]
    y = [3, 4]
    fig = create_scatter_plot(x, y, "Test", marker_color="red")
    assert fig.data[0].marker.color == "red"


def test_custom_marker_size():
    """Should accept custom marker size"""
    x = [1, 2]
    y = [3, 4]
    fig = create_scatter_plot(x, y, "Test", marker_size=15)
    assert fig.data[0].marker.size == 15


def test_empty_data():
    """Should handle empty data lists"""
    x = []
    y = []
    fig = create_scatter_plot(x, y, "Empty")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 0


def test_single_point():
    """Should handle single data point"""
    x = [5.0]
    y = [10.0]
    fig = create_scatter_plot(x, y, "Single")
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y
