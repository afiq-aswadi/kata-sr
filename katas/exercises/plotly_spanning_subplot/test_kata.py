"""Tests for plotly_spanning_subplot kata."""

import plotly.graph_objects as go
import pytest

try:
    from user_kata import create_spanning_subplot
except ImportError:
    from .reference import create_spanning_subplot


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    top = [{"x": [1, 2], "y": [3, 4]}, {"x": [5, 6], "y": [7, 8]}]
    bottom = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_spanning_subplot(top, bottom)
    assert isinstance(fig, go.Figure)


def test_has_three_traces():
    """Figure should have 3 traces (2 scatter + 1 heatmap)."""
    top = [{"x": [1, 2], "y": [3, 4]}, {"x": [5, 6], "y": [7, 8]}]
    bottom = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_spanning_subplot(top, bottom)
    assert len(fig.data) == 3


def test_trace_types():
    """Should have 2 Scatter traces and 1 Heatmap trace."""
    top = [{"x": [1, 2], "y": [3, 4]}, {"x": [5, 6], "y": [7, 8]}]
    bottom = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_spanning_subplot(top, bottom)

    assert isinstance(fig.data[0], go.Scatter)
    assert isinstance(fig.data[1], go.Scatter)
    assert isinstance(fig.data[2], go.Heatmap)


def test_top_data_preserved():
    """Top scatter plots should have correct data."""
    top = [{"x": [1, 2, 3], "y": [10, 20, 30]}, {"x": [4, 5, 6], "y": [40, 50, 60]}]
    bottom = {"z": [[1, 2], [3, 4]]}

    fig = create_spanning_subplot(top, bottom)

    # Check top-left scatter
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [10, 20, 30]

    # Check top-right scatter
    assert list(fig.data[1].x) == [4, 5, 6]
    assert list(fig.data[1].y) == [40, 50, 60]


def test_bottom_data_preserved():
    """Bottom heatmap should have correct data."""
    top = [{"x": [1, 2], "y": [3, 4]}, {"x": [5, 6], "y": [7, 8]}]
    bottom = {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}

    fig = create_spanning_subplot(top, bottom)

    # Check heatmap data (plotly returns tuples)
    expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list(map(list, fig.data[2].z)) == expected


def test_three_axes_pairs():
    """Layout should have 3 pairs of axes (2 for top, 1 for bottom)."""
    top = [{"x": [1, 2], "y": [3, 4]}, {"x": [5, 6], "y": [7, 8]}]
    bottom = {"z": [[1, 2], [3, 4]]}

    fig = create_spanning_subplot(top, bottom)

    # Should have xaxis, xaxis2, xaxis3
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout
    assert "xaxis3" in fig.layout

    # Should have yaxis, yaxis2, yaxis3
    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert "yaxis3" in fig.layout


def test_empty_data():
    """Should handle empty data arrays."""
    top = [{"x": [], "y": []}, {"x": [], "y": []}]
    bottom = {"z": []}

    fig = create_spanning_subplot(top, bottom)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_single_element_data():
    """Should handle single data points."""
    top = [{"x": [1], "y": [2]}, {"x": [3], "y": [4]}]
    bottom = {"z": [[5]]}

    fig = create_spanning_subplot(top, bottom)

    assert len(fig.data) == 3
    assert list(fig.data[0].x) == [1]
    assert list(fig.data[1].x) == [3]
