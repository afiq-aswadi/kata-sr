"""Tests for plotly_bar kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_bar_chart
except ImportError:
    from .reference import create_bar_chart


def test_returns_figure():
    """Should return a go.Figure object"""
    categories = ["A", "B", "C"]
    values = [10, 20, 15]
    fig = create_bar_chart(categories, values, "Test")
    assert isinstance(fig, go.Figure)


def test_has_bar_trace():
    """Figure should contain a bar trace"""
    categories = ["A", "B"]
    values = [5, 10]
    fig = create_bar_chart(categories, values, "Test")
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_correct_categories():
    """Bar chart should have correct category labels"""
    categories = ["Q1", "Q2", "Q3", "Q4"]
    values = [100, 150, 125, 175]
    fig = create_bar_chart(categories, values, "Test")
    assert list(fig.data[0].x) == categories


def test_correct_values():
    """Bar chart should have correct bar heights"""
    categories = ["A", "B", "C"]
    values = [10, 25, 15]
    fig = create_bar_chart(categories, values, "Test")
    assert list(fig.data[0].y) == values


def test_title_set():
    """Figure should have the specified title"""
    categories = ["X", "Y"]
    values = [1, 2]
    title = "Sales by Region"
    fig = create_bar_chart(categories, values, title)
    assert fig.layout.title.text == title


def test_single_bar():
    """Should handle single category"""
    categories = ["Only"]
    values = [42]
    fig = create_bar_chart(categories, values, "Single")
    assert list(fig.data[0].x) == categories
    assert list(fig.data[0].y) == values


def test_many_categories():
    """Should handle many categories"""
    categories = [f"Cat{i}" for i in range(20)]
    values = list(range(20))
    fig = create_bar_chart(categories, values, "Many")
    assert len(fig.data[0].x) == 20


def test_zero_values():
    """Should handle zero values"""
    categories = ["A", "B", "C"]
    values = [0, 10, 0]
    fig = create_bar_chart(categories, values, "Zeros")
    assert list(fig.data[0].y) == values


def test_negative_values():
    """Should handle negative values"""
    categories = ["A", "B", "C"]
    values = [-5, 10, -3]
    fig = create_bar_chart(categories, values, "Mixed")
    assert list(fig.data[0].y) == values
