"""Tests for plotly_update_subplot_axes kata."""

import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

try:
    from user_kata import update_subplot_axes
except ImportError:
    from .reference import update_subplot_axes


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(fig, 1, 1, x_title="Time")
    assert isinstance(result, go.Figure)


def test_x_title_updated():
    """Should update x-axis title for specified subplot."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(fig, 1, 1, x_title="Time (s)")

    assert result.layout.xaxis.title.text == "Time (s)"


def test_y_title_updated():
    """Should update y-axis title for specified subplot."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(fig, 1, 1, y_title="Value")

    assert result.layout.yaxis.title.text == "Value"


def test_x_range_updated():
    """Should update x-axis range for specified subplot."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(fig, 1, 1, x_range=[0, 100])

    assert list(result.layout.xaxis.range) == [0, 100]


def test_y_range_updated():
    """Should update y-axis range for specified subplot."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(fig, 1, 1, y_range=[-10, 10])

    assert list(result.layout.yaxis.range) == [-10, 10]


def test_multiple_properties():
    """Should update multiple properties simultaneously."""
    fig = make_subplots(rows=2, cols=2)
    result = update_subplot_axes(
        fig,
        1,
        1,
        x_title="Time",
        y_title="Amplitude",
        x_range=[0, 5],
        y_range=[-1, 1],
    )

    assert result.layout.xaxis.title.text == "Time"
    assert result.layout.yaxis.title.text == "Amplitude"
    assert list(result.layout.xaxis.range) == [0, 5]
    assert list(result.layout.yaxis.range) == [-1, 1]


def test_different_subplots():
    """Should target the correct subplot based on row/col."""
    fig = make_subplots(rows=2, cols=2)

    # Update subplot (1,2)
    result = update_subplot_axes(fig, 1, 2, x_title="X2", y_title="Y2")

    # Check that xaxis2 and yaxis2 are updated (subplot 1,2)
    assert result.layout.xaxis2.title.text == "X2"
    assert result.layout.yaxis2.title.text == "Y2"


def test_none_values_ignored():
    """Should only update provided (non-None) properties."""
    fig = make_subplots(rows=2, cols=2)

    # Set initial value
    result = update_subplot_axes(fig, 1, 1, x_title="Initial")

    # Update only y_title (x_title should remain)
    result = update_subplot_axes(result, 1, 1, y_title="Y-value")

    # x_title should still be "Initial"
    assert result.layout.xaxis.title.text == "Initial"
    assert result.layout.yaxis.title.text == "Y-value"


def test_chaining_updates():
    """Should support chaining multiple update calls."""
    fig = make_subplots(rows=2, cols=2)

    result = update_subplot_axes(fig, 1, 1, x_title="X1")
    result = update_subplot_axes(result, 1, 2, x_title="X2")
    result = update_subplot_axes(result, 2, 1, x_title="X3")

    assert result.layout.xaxis.title.text == "X1"
    assert result.layout.xaxis2.title.text == "X2"
    assert result.layout.xaxis3.title.text == "X3"
