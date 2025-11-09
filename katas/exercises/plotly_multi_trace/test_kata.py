"""Tests for plotly_multi_trace kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_multi_trace_plot
except ImportError:
    from .reference import create_multi_trace_plot


def test_returns_figure():
    """Should return a go.Figure object"""
    x = [1, 2, 3]
    y1 = [1, 2, 3]
    y2 = [1.5, 2.5, 3.5]
    fig = create_multi_trace_plot(x, y1, y2)
    assert isinstance(fig, go.Figure)


def test_has_two_traces():
    """Figure should contain exactly two traces"""
    x = [1, 2, 3]
    y1 = [1, 2, 3]
    y2 = [1, 2, 3]
    fig = create_multi_trace_plot(x, y1, y2)
    assert len(fig.data) == 2


def test_first_trace_is_scatter():
    """First trace should be scatter with markers"""
    x = [1, 2, 3]
    y1 = [1, 2, 3]
    y2 = [1, 2, 3]
    fig = create_multi_trace_plot(x, y1, y2)
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"


def test_second_trace_is_line():
    """Second trace should be line"""
    x = [1, 2, 3]
    y1 = [1, 2, 3]
    y2 = [1, 2, 3]
    fig = create_multi_trace_plot(x, y1, y2)
    assert fig.data[1].type == "scatter"
    assert fig.data[1].mode == "lines"


def test_scatter_trace_data():
    """Scatter trace should have correct data"""
    x = [1, 2, 3, 4]
    y_scatter = [10, 20, 15, 25]
    y_line = [10, 15, 20, 25]
    fig = create_multi_trace_plot(x, y_scatter, y_line)
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y_scatter


def test_line_trace_data():
    """Line trace should have correct data"""
    x = [1, 2, 3, 4]
    y_scatter = [10, 20, 15, 25]
    y_line = [10, 15, 20, 25]
    fig = create_multi_trace_plot(x, y_scatter, y_line)
    assert list(fig.data[1].x) == x
    assert list(fig.data[1].y) == y_line


def test_scatter_styling():
    """Scatter trace should have blue markers"""
    x = [1, 2]
    y1 = [1, 2]
    y2 = [1, 2]
    fig = create_multi_trace_plot(x, y1, y2)
    assert fig.data[0].marker.color == "blue"


def test_line_styling():
    """Line trace should be red and dashed"""
    x = [1, 2]
    y1 = [1, 2]
    y2 = [1, 2]
    fig = create_multi_trace_plot(x, y1, y2)
    assert fig.data[1].line.color == "red"
    assert fig.data[1].line.dash == "dash"


def test_trace_names():
    """Traces should have proper names for legend"""
    x = [1, 2]
    y1 = [1, 2]
    y2 = [1, 2]
    fig = create_multi_trace_plot(x, y1, y2)
    assert fig.data[0].name == "Data Points"
    assert fig.data[1].name == "Trend"
