"""Tests for plotly_layout kata."""

import plotly.graph_objects as go

try:
    from user_kata import configure_layout
except ImportError:
    from .reference import configure_layout


def test_returns_figure():
    """Should return a go.Figure object"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    result = configure_layout(fig, "Title", "X", "Y")
    assert isinstance(result, go.Figure)


def test_sets_title():
    """Should set the main title"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    title = "My Plot Title"
    result = configure_layout(fig, title, "X", "Y")
    assert result.layout.title.text == title


def test_sets_x_axis_label():
    """Should set x-axis label"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    x_label = "Time (seconds)"
    result = configure_layout(fig, "Title", x_label, "Y")
    assert result.layout.xaxis.title.text == x_label


def test_sets_y_axis_label():
    """Should set y-axis label"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    y_label = "Temperature (°C)"
    result = configure_layout(fig, "Title", "X", y_label)
    assert result.layout.yaxis.title.text == y_label


def test_shows_legend_by_default():
    """Should show legend by default"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="Series 1"))
    result = configure_layout(fig, "Title", "X", "Y")
    assert result.layout.showlegend is True


def test_hides_legend_when_false():
    """Should hide legend when showlegend=False"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="Series 1"))
    result = configure_layout(fig, "Title", "X", "Y", showlegend=False)
    assert result.layout.showlegend is False


def test_modifies_existing_figure():
    """Should modify the original figure (not create new one)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    result = configure_layout(fig, "New Title", "X", "Y")
    # Should be same object
    assert result is fig


def test_preserves_traces():
    """Should preserve existing traces"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Data"))
    result = configure_layout(fig, "Title", "X", "Y")
    assert len(result.data) == 1
    assert list(result.data[0].x) == [1, 2, 3]


def test_all_parameters_together():
    """Should correctly apply all layout parameters"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["A", "B"], y=[1, 2]))
    result = configure_layout(
        fig,
        title="Complete Layout",
        x_label="Categories",
        y_label="Values",
        showlegend=False
    )
    assert result.layout.title.text == "Complete Layout"
    assert result.layout.xaxis.title.text == "Categories"
    assert result.layout.yaxis.title.text == "Values"
    assert result.layout.showlegend is False
