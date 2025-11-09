"""Tests for plotly_shared_axes kata."""

import plotly.graph_objects as go
import pytest

try:
    from user_kata import create_shared_axes_plot
except ImportError:
    from .reference import create_shared_axes_plot


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
    ]
    fig = create_shared_axes_plot(data)
    assert isinstance(fig, go.Figure)


def test_correct_number_of_traces():
    """Should create one trace per time series."""
    data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
        {"x": [1, 2, 3], "y": [8, 12, 18], "name": "Series 3"},
    ]
    fig = create_shared_axes_plot(data)
    assert len(fig.data) == 3


def test_all_traces_are_scatter():
    """All traces should be Scatter (line) type."""
    data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
    ]
    fig = create_shared_axes_plot(data)

    for trace in fig.data:
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"


def test_data_preserved():
    """Input data should be correctly assigned to traces."""
    data = [
        {"x": [1, 2, 3, 4], "y": [10, 20, 15, 25], "name": "Series 1"},
        {"x": [1, 2, 3, 4], "y": [5, 15, 10, 20], "name": "Series 2"},
    ]
    fig = create_shared_axes_plot(data)

    assert list(fig.data[0].x) == [1, 2, 3, 4]
    assert list(fig.data[0].y) == [10, 20, 15, 25]
    assert fig.data[0].name == "Series 1"

    assert list(fig.data[1].x) == [1, 2, 3, 4]
    assert list(fig.data[1].y) == [5, 15, 10, 20]
    assert fig.data[1].name == "Series 2"


def test_multiple_axes_created():
    """Layout should have separate y-axes for each subplot."""
    data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
        {"x": [1, 2, 3], "y": [8, 12, 18], "name": "Series 3"},
    ]
    fig = create_shared_axes_plot(data)

    # Should have 3 x-axes and 3 y-axes
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout
    assert "xaxis3" in fig.layout

    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert "yaxis3" in fig.layout


def test_shared_axes_configured():
    """X-axes should be configured to share (via matches attribute)."""
    data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
        {"x": [1, 2, 3], "y": [8, 12, 18], "name": "Series 3"},
    ]
    fig = create_shared_axes_plot(data)

    # Check for shared axes configuration
    # At least one x-axis should have matches attribute set
    has_shared_config = False
    for i in range(1, 4):
        if i == 1:
            axis_name = "xaxis"
        else:
            axis_name = f"xaxis{i}"

        if axis_name in fig.layout:
            axis = fig.layout[axis_name]
            if hasattr(axis, "matches") and axis.matches is not None and axis.matches != "":
                has_shared_config = True
                break

    assert has_shared_config, "X-axes should be configured to share"


def test_single_series():
    """Should handle a single time series."""
    data = [{"x": [1, 2, 3], "y": [10, 20, 15], "name": "Only Series"}]
    fig = create_shared_axes_plot(data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_empty_data():
    """Should handle empty data lists."""
    data = [
        {"x": [], "y": [], "name": "Empty 1"},
        {"x": [], "y": [], "name": "Empty 2"},
    ]
    fig = create_shared_axes_plot(data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
