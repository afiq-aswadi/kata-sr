"""Tests for plotly_hover_template kata."""

import plotly.graph_objects as go

try:
    from user_kata import add_hover_template
except ImportError:
    from .reference import add_hover_template


def test_returns_figure():
    """Should return a go.Figure object"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    template = "Value: %{y}"
    result = add_hover_template(fig, template)
    assert isinstance(result, go.Figure)


def test_sets_hover_template():
    """Should set hovertemplate on trace"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    template = "X: %{x}<br>Y: %{y}"
    result = add_hover_template(fig, template)
    assert result.data[0].hovertemplate == template


def test_simple_template():
    """Should handle simple hover template"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 20, 30]))
    template = "Value: %{y}<extra></extra>"
    result = add_hover_template(fig, template)
    assert result.data[0].hovertemplate == template


def test_formatted_template():
    """Should handle formatted hover template"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1.234, 5.678]))
    template = "Y: %{y:.2f}"
    result = add_hover_template(fig, template)
    assert result.data[0].hovertemplate == template


def test_multiple_traces():
    """Should update all traces"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="A"))
    fig.add_trace(go.Scatter(x=[1, 2], y=[5, 6], name="B"))
    template = "Custom: %{y}"
    result = add_hover_template(fig, template)
    assert result.data[0].hovertemplate == template
    assert result.data[1].hovertemplate == template


def test_preserves_data():
    """Should preserve trace data"""
    fig = go.Figure()
    x = [1, 2, 3]
    y = [10, 20, 30]
    fig.add_trace(go.Scatter(x=x, y=y))
    template = "Test: %{y}"
    result = add_hover_template(fig, template)
    assert list(result.data[0].x) == x
    assert list(result.data[0].y) == y


def test_modifies_existing_figure():
    """Should modify original figure"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    template = "Modified"
    result = add_hover_template(fig, template)
    assert result is fig


def test_bar_chart():
    """Should work with bar charts"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["A", "B"], y=[10, 20]))
    template = "Category: %{x}<br>Value: %{y}"
    result = add_hover_template(fig, template)
    assert result.data[0].hovertemplate == template
