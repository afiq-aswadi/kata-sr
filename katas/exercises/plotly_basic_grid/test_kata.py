"""Tests for plotly_basic_grid kata."""

import plotly.graph_objects as go
import pytest

try:
    from user_kata import create_subplot_grid
except ImportError:
    from .reference import create_subplot_grid


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)
    assert isinstance(fig, go.Figure)


def test_has_four_traces():
    """Grid should contain exactly 4 traces (one per subplot)."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)
    assert len(fig.data) == 4


def test_trace_types():
    """Each trace should have the correct type."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)

    # Check trace types
    assert isinstance(fig.data[0], go.Scatter)  # Scatter
    assert isinstance(fig.data[1], go.Bar)  # Bar
    assert isinstance(fig.data[2], go.Scatter)  # Line (also Scatter type)
    assert isinstance(fig.data[3], go.Heatmap)  # Heatmap


def test_scatter_mode():
    """Scatter plot should have markers mode."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)
    assert fig.data[0].mode == "markers"


def test_line_mode():
    """Line plot should have lines mode."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)
    assert fig.data[2].mode == "lines"


def test_data_preserved():
    """Input data should be correctly assigned to traces."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)

    # Check scatter data
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [4, 5, 6]

    # Check bar data
    assert list(fig.data[1].x) == ["A", "B", "C"]
    assert list(fig.data[1].y) == [10, 20, 30]


def test_grid_axes_created():
    """Layout should have 4 pairs of axes for 2x2 grid."""
    scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
    bar = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
    line = {"x": [0, 1, 2], "y": [1, 4, 9]}
    heatmap = {"z": [[1, 2, 3], [4, 5, 6]]}

    fig = create_subplot_grid(scatter, bar, line, heatmap)

    # Check that all 4 x and y axes exist
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout
    assert "xaxis3" in fig.layout
    assert "xaxis4" in fig.layout

    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert "yaxis3" in fig.layout
    assert "yaxis4" in fig.layout


def test_empty_lists():
    """Should handle empty data lists."""
    scatter = {"x": [], "y": []}
    bar = {"x": [], "y": []}
    line = {"x": [], "y": []}
    heatmap = {"z": []}

    fig = create_subplot_grid(scatter, bar, line, heatmap)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
