"""Tests for plotly_line kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_line_plot
except ImportError:
    from .reference import create_line_plot


def test_returns_figure():
    """Should return a go.Figure object"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_line_plot(x, y, "Test Line")
    assert isinstance(fig, go.Figure)


def test_has_scatter_trace():
    """Figure should contain a scatter trace (lines use Scatter)"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_line_plot(x, y, "Test")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_mode_is_lines():
    """Scatter trace should use lines mode"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_line_plot(x, y, "Test")
    assert fig.data[0].mode == "lines"


def test_correct_data():
    """Trace should contain the correct x and y data"""
    x = [1, 2, 3, 4]
    y = [10, 20, 15, 25]
    fig = create_line_plot(x, y, "Test")
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_line_name_set():
    """Line should have the specified name"""
    x = [1, 2]
    y = [3, 4]
    name = "My Line"
    fig = create_line_plot(x, y, name)
    assert fig.data[0].name == name


def test_default_line_color():
    """Should use default blue color"""
    x = [1, 2]
    y = [3, 4]
    fig = create_line_plot(x, y, "Test")
    assert fig.data[0].line.color == "blue"


def test_custom_line_color():
    """Should accept custom line color"""
    x = [1, 2]
    y = [3, 4]
    fig = create_line_plot(x, y, "Test", line_color="red")
    assert fig.data[0].line.color == "red"


def test_single_segment():
    """Should handle two points (single line segment)"""
    x = [0, 1]
    y = [0, 1]
    fig = create_line_plot(x, y, "Diagonal")
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_many_points():
    """Should handle many data points"""
    x = list(range(100))
    y = [i**2 for i in x]
    fig = create_line_plot(x, y, "Parabola")
    assert len(fig.data[0].x) == 100
