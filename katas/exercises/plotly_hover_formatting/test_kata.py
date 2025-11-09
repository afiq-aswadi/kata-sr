"""Tests for plotly_hover_formatting kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_formatted_hover_scatter
except ImportError:
    from .reference import create_formatted_hover_scatter


def test_returns_figure():
    """Should return a go.Figure object"""
    x = [1, 2, 3]
    y = [1.1, 2.2, 3.3]
    labels = ["A", "B", "C"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert isinstance(fig, go.Figure)


def test_has_scatter_trace():
    """Should have scatter trace"""
    x = [1, 2]
    y = [1.5, 2.5]
    labels = ["Point 1", "Point 2"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"


def test_correct_data():
    """Should have correct x and y data"""
    x = [1, 2, 3, 4]
    y = [10.5, 20.3, 15.7, 25.1]
    labels = ["A", "B", "C", "D"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_text_parameter_set():
    """Should set text parameter with labels"""
    x = [1, 2, 3]
    y = [1, 2, 3]
    labels = ["First", "Second", "Third"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert list(fig.data[0].text) == labels


def test_has_hovertemplate():
    """Should have custom hovertemplate"""
    x = [1, 2]
    y = [1.5, 2.5]
    labels = ["A", "B"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert fig.data[0].hovertemplate is not None


def test_template_includes_text():
    """Hovertemplate should include %{text}"""
    x = [1]
    y = [1]
    labels = ["Test"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert "%{text}" in fig.data[0].hovertemplate


def test_template_includes_x():
    """Hovertemplate should include %{x}"""
    x = [1]
    y = [1]
    labels = ["Test"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert "%{x}" in fig.data[0].hovertemplate


def test_template_includes_formatted_y():
    """Hovertemplate should include formatted y (:.2f)"""
    x = [1]
    y = [1.234]
    labels = ["Test"]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert "%{y:.2f}" in fig.data[0].hovertemplate


def test_many_points():
    """Should handle many labeled points"""
    n = 50
    x = list(range(n))
    y = [float(i) * 1.5 for i in x]
    labels = [f"Point {i}" for i in range(n)]
    fig = create_formatted_hover_scatter(x, y, labels)
    assert len(fig.data[0].x) == n
    assert len(fig.data[0].text) == n
