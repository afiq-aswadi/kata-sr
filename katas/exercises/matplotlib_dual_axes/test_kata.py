"""Tests for matplotlib dual y-axes kata."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) * 10 + 50  # Range: 40-60
    y2 = np.cos(x) * 30 + 50  # Range: 20-80
    return x, y1, y2


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


def test_returns_three_objects(sample_data):
    """Test that function returns fig, ax1, ax2."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    result = create_dual_axis_plot(x, y1, y2)

    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 3, "Should return (fig, ax1, ax2)"


def test_creates_figure(sample_data):
    """Test that a figure is created."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    assert isinstance(fig, plt.Figure), "First element should be a Figure"


def test_creates_two_axes(sample_data):
    """Test that two axes objects are created."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    assert isinstance(ax1, plt.Axes), "Second element should be an Axes"
    assert isinstance(ax2, plt.Axes), "Third element should be an Axes"
    assert ax1 is not ax2, "The two axes should be different objects"


def test_both_axes_have_data(sample_data):
    """Test that both axes have plotted data."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    assert len(ax1.get_lines()) > 0, "Primary axis should have plotted data"
    assert len(ax2.get_lines()) > 0, "Secondary axis should have plotted data"


def test_axes_have_labels(sample_data):
    """Test that both axes have y-labels."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    assert ax1.get_ylabel() != "", "Primary axis should have y-label"
    assert ax2.get_ylabel() != "", "Secondary axis should have y-label"


def test_labels_have_different_colors(sample_data):
    """Test that y-axis labels have different colors."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    y1_label_color = ax1.yaxis.label.get_color()
    y2_label_color = ax2.yaxis.label.get_color()

    assert y1_label_color != y2_label_color, "Y-axis labels should have different colors"


def test_uses_blue_and_red_colors(sample_data):
    """Test that the plot uses blue and red for the two axes."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    # Check that one axis is blueish and one is reddish
    y1_label_color = ax1.yaxis.label.get_color()
    y2_label_color = ax2.yaxis.label.get_color()

    # Convert to lowercase for comparison
    y1_str = str(y1_label_color).lower()
    y2_str = str(y2_label_color).lower()

    has_blue = "blue" in y1_str or "blue" in y2_str
    has_red = "red" in y1_str or "red" in y2_str

    assert has_blue, "One axis should use blue color"
    assert has_red, "One axis should use red color"


def test_handles_different_scales(sample_data):
    """Test that dual axes can handle very different scales."""
    from template import create_dual_axis_plot

    x = np.linspace(0, 10, 50)
    y1 = np.array([1, 2, 3, 4, 5] * 10)  # Small scale
    y2 = np.array([1000, 2000, 3000, 4000, 5000] * 10)  # Large scale

    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    # Both should plot successfully despite different scales
    assert len(ax1.get_lines()) > 0, "Should handle small scale data"
    assert len(ax2.get_lines()) > 0, "Should handle large scale data"


def test_axes_share_x_axis(sample_data):
    """Test that both axes share the same x-axis data."""
    from template import create_dual_axis_plot

    x, y1, y2 = sample_data
    fig, ax1, ax2 = create_dual_axis_plot(x, y1, y2)

    # Get x data from both plots
    line1_xdata = ax1.get_lines()[0].get_xdata()
    line2_xdata = ax2.get_lines()[0].get_xdata()

    assert np.array_equal(line1_xdata, line2_xdata), "Both axes should share x data"
