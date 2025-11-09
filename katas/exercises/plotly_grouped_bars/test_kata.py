"""Tests for plotly_grouped_bars kata."""

import plotly.graph_objects as go

try:
    from user_kata import create_grouped_bar_chart
except ImportError:
    from .reference import create_grouped_bar_chart


def test_returns_figure():
    """Should return a go.Figure object"""
    categories = ["A", "B"]
    g1 = [10, 20]
    g2 = [15, 25]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert isinstance(fig, go.Figure)


def test_has_two_bar_traces():
    """Should have exactly two bar traces"""
    categories = ["X", "Y", "Z"]
    g1 = [1, 2, 3]
    g2 = [4, 5, 6]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "bar"


def test_first_group_data():
    """First bar trace should have correct categories and values"""
    categories = ["Q1", "Q2", "Q3"]
    g1 = [100, 150, 125]
    g2 = [110, 140, 130]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert list(fig.data[0].x) == categories
    assert list(fig.data[0].y) == g1


def test_second_group_data():
    """Second bar trace should have correct categories and values"""
    categories = ["Q1", "Q2", "Q3"]
    g1 = [100, 150, 125]
    g2 = [110, 140, 130]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert list(fig.data[1].x) == categories
    assert list(fig.data[1].y) == g2


def test_default_group_names():
    """Should use default group names"""
    categories = ["A"]
    g1 = [10]
    g2 = [20]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert fig.data[0].name == "Group 1"
    assert fig.data[1].name == "Group 2"


def test_custom_group_names():
    """Should accept custom group names"""
    categories = ["A", "B"]
    g1 = [10, 20]
    g2 = [15, 25]
    fig = create_grouped_bar_chart(
        categories, g1, g2, group1_name="Sales", group2_name="Costs"
    )
    assert fig.data[0].name == "Sales"
    assert fig.data[1].name == "Costs"


def test_many_categories():
    """Should handle many categories"""
    categories = [f"Cat{i}" for i in range(10)]
    g1 = list(range(10))
    g2 = list(range(10, 20))
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert len(fig.data[0].x) == 10
    assert len(fig.data[1].x) == 10


def test_zero_values():
    """Should handle zero values"""
    categories = ["A", "B", "C"]
    g1 = [0, 10, 0]
    g2 = [5, 0, 5]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert list(fig.data[0].y) == g1
    assert list(fig.data[1].y) == g2


def test_negative_values():
    """Should handle negative values"""
    categories = ["Jan", "Feb", "Mar"]
    g1 = [10, -5, 15]
    g2 = [-3, 8, -2]
    fig = create_grouped_bar_chart(categories, g1, g2)
    assert list(fig.data[0].y) == g1
    assert list(fig.data[1].y) == g2
