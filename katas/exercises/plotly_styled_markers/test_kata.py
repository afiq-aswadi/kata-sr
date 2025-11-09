"""Tests for plotly_styled_markers kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_styled_scatter
except ImportError:
    from .reference import create_styled_scatter


def test_returns_figure():
    """Should return a go.Figure object"""
    x = [1, 2, 3]
    y = [1, 2, 3]
    fig = create_styled_scatter(x, y)
    assert isinstance(fig, go.Figure)


def test_has_scatter_trace():
    """Should have scatter trace with markers"""
    x = [1, 2]
    y = [3, 4]
    fig = create_styled_scatter(x, y)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"


def test_without_styling():
    """Should work without colors or sizes"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    fig = create_styled_scatter(x, y)
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_with_colors():
    """Should apply custom colors to markers"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    colors = ["red", "blue", "green"]
    fig = create_styled_scatter(x, y, colors=colors)
    assert list(fig.data[0].marker.color) == colors


def test_with_sizes():
    """Should apply custom sizes to markers"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    sizes = [10, 20, 15]
    fig = create_styled_scatter(x, y, sizes=sizes)
    assert list(fig.data[0].marker.size) == sizes


def test_with_colors_and_sizes():
    """Should apply both colors and sizes"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    colors = ["red", "blue", "green"]
    sizes = [8, 12, 16]
    fig = create_styled_scatter(x, y, colors=colors, sizes=sizes)
    assert list(fig.data[0].marker.color) == colors
    assert list(fig.data[0].marker.size) == sizes


def test_single_point_with_styling():
    """Should handle single point with styling"""
    x = [5]
    y = [10]
    colors = ["purple"]
    sizes = [20]
    fig = create_styled_scatter(x, y, colors=colors, sizes=sizes)
    assert list(fig.data[0].marker.color) == colors
    assert list(fig.data[0].marker.size) == sizes


def test_data_integrity():
    """Should preserve x and y data exactly"""
    x = [1.5, 2.5, 3.5]
    y = [100.1, 200.2, 150.3]
    colors = ["red", "green", "blue"]
    fig = create_styled_scatter(x, y, colors=colors)
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y
