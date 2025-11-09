"""Tests for plotly_dual_yaxis kata."""

import plotly.graph_objects as go
import pytest

try:
    from user_kata import create_dual_yaxis_plot
except ImportError:
    from .reference import create_dual_yaxis_plot


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    primary = {"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "name": "Temperature"}
    secondary = {"x": [1, 2, 3, 4], "y": [100, 200, 150, 250], "name": "Pressure"}

    fig = create_dual_yaxis_plot(primary, secondary)
    assert isinstance(fig, go.Figure)


def test_has_two_traces():
    """Figure should have exactly 2 traces."""
    primary = {"x": [1, 2, 3], "y": [10, 20, 30], "name": "Primary"}
    secondary = {"x": [1, 2, 3], "y": [100, 200, 300], "name": "Secondary"}

    fig = create_dual_yaxis_plot(primary, secondary)
    assert len(fig.data) == 2


def test_both_scatter_traces():
    """Both traces should be Scatter type."""
    primary = {"x": [1, 2, 3], "y": [10, 20, 30], "name": "Primary"}
    secondary = {"x": [1, 2, 3], "y": [100, 200, 300], "name": "Secondary"}

    fig = create_dual_yaxis_plot(primary, secondary)

    assert isinstance(fig.data[0], go.Scatter)
    assert isinstance(fig.data[1], go.Scatter)


def test_data_preserved():
    """Input data should be correctly assigned to traces."""
    primary = {"x": [1, 2, 3, 4], "y": [10, 15, 13, 17], "name": "Primary"}
    secondary = {"x": [1, 2, 3, 4], "y": [100, 150, 130, 170], "name": "Secondary"}

    fig = create_dual_yaxis_plot(primary, secondary)

    # Check primary data
    assert list(fig.data[0].x) == [1, 2, 3, 4]
    assert list(fig.data[0].y) == [10, 15, 13, 17]
    assert fig.data[0].name == "Primary"

    # Check secondary data
    assert list(fig.data[1].x) == [1, 2, 3, 4]
    assert list(fig.data[1].y) == [100, 150, 130, 170]
    assert fig.data[1].name == "Secondary"


def test_dual_yaxes_exist():
    """Layout should have both yaxis and yaxis2."""
    primary = {"x": [1, 2, 3], "y": [10, 20, 30], "name": "Temperature"}
    secondary = {"x": [1, 2, 3], "y": [100, 200, 300], "name": "Pressure"}

    fig = create_dual_yaxis_plot(primary, secondary)

    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout


def test_yaxis_titles():
    """Y-axis titles should match trace names."""
    primary = {"x": [1, 2, 3], "y": [10, 20, 30], "name": "Temperature"}
    secondary = {"x": [1, 2, 3], "y": [100, 200, 300], "name": "Pressure"}

    fig = create_dual_yaxis_plot(primary, secondary)

    assert fig.layout.yaxis.title.text == "Temperature"
    assert fig.layout.yaxis2.title.text == "Pressure"


def test_different_scales():
    """Should handle vastly different y-axis scales."""
    primary = {"x": [1, 2, 3], "y": [1, 2, 3], "name": "Small"}
    secondary = {"x": [1, 2, 3], "y": [1000, 2000, 3000], "name": "Large"}

    fig = create_dual_yaxis_plot(primary, secondary)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_single_point():
    """Should handle single data point."""
    primary = {"x": [1], "y": [10], "name": "Point 1"}
    secondary = {"x": [1], "y": [100], "name": "Point 2"}

    fig = create_dual_yaxis_plot(primary, secondary)

    assert len(fig.data) == 2
    assert len(fig.data[0].x) == 1
    assert len(fig.data[1].x) == 1
