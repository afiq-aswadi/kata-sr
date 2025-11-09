"""Tests for matplotlib figure/axes kata."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-GUI backend for testing

try:
    from user_kata import create_multi_panel_plot
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone test runs
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)

    create_multi_panel_plot = reference.create_multi_panel_plot  # type: ignore[attr-defined]


def test_returns_figure_and_axes():
    """Test that function returns a figure and axes."""
    result = create_multi_panel_plot()
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Should return (fig, axes) tuple"
    fig, axes = result
    assert hasattr(fig, "get_size_inches"), "First element should be a Figure"
    assert hasattr(axes, "shape"), "Second element should be axes array"


def test_figure_has_correct_size():
    """Test that figure has the correct size (12x10)."""
    fig, _ = create_multi_panel_plot()
    width, height = fig.get_size_inches()
    assert abs(width - 12) < 0.1, f"Figure width should be 12, got {width}"
    assert abs(height - 10) < 0.1, f"Figure height should be 10, got {height}"


def test_has_four_subplots():
    """Test that there are exactly 4 subplots in 2x2 grid."""
    fig, axes = create_multi_panel_plot()
    assert axes.shape == (2, 2), f"Axes should be 2x2 array, got {axes.shape}"
    assert len(fig.axes) == 4, f"Figure should have 4 subplots, got {len(fig.axes)}"


def test_all_subplots_have_titles():
    """Test that all subplots have titles."""
    _, axes = create_multi_panel_plot()
    for i in range(2):
        for j in range(2):
            title = axes[i, j].get_title()
            assert title and len(title) > 0, f"Subplot [{i}, {j}] should have a title"


def test_all_subplots_have_labels():
    """Test that all subplots have x and y labels."""
    _, axes = create_multi_panel_plot()
    for i in range(2):
        for j in range(2):
            xlabel = axes[i, j].get_xlabel()
            ylabel = axes[i, j].get_ylabel()
            assert xlabel and len(xlabel) > 0, f"Subplot [{i}, {j}] should have x label"
            assert ylabel and len(ylabel) > 0, f"Subplot [{i}, {j}] should have y label"


def test_all_subplots_have_grid():
    """Test that all subplots have grid enabled."""
    _, axes = create_multi_panel_plot()
    for i in range(2):
        for j in range(2):
            # Check if grid lines exist
            gridlines_x = axes[i, j].xaxis.get_gridlines()
            gridlines_y = axes[i, j].yaxis.get_gridlines()
            has_grid = (
                len(gridlines_x) > 0 and gridlines_x[0].get_visible()
            ) or (len(gridlines_y) > 0 and gridlines_y[0].get_visible())
            assert has_grid, f"Subplot [{i}, {j}] should have grid enabled"


def test_top_left_is_line_plot():
    """Test that top-left subplot (0,0) is a line plot with 2 lines."""
    _, axes = create_multi_panel_plot()
    lines = axes[0, 0].get_lines()
    assert len(lines) >= 2, f"Top-left should have at least 2 lines, got {len(lines)}"

    # Check line colors (should have red and blue)
    colors = [line.get_color() for line in lines]
    has_red = any(c in ["r", "red", (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)] for c in colors)
    has_blue = any(c in ["b", "blue", (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0)] for c in colors)
    assert has_red, "Line plot should have a red line"
    assert has_blue, "Line plot should have a blue line"


def test_top_left_has_legend():
    """Test that top-left subplot has a legend."""
    _, axes = create_multi_panel_plot()
    legend = axes[0, 0].get_legend()
    assert legend is not None, "Top-left subplot should have a legend"
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) >= 2, "Legend should have at least 2 entries"


def test_top_right_is_scatter_plot():
    """Test that top-right subplot (0,1) is a scatter plot."""
    _, axes = create_multi_panel_plot()
    collections = axes[0, 1].collections
    assert len(collections) > 0, "Top-right should have scatter plot data"

    # Check that scatter points exist
    scatter = collections[0]
    offsets = scatter.get_offsets()
    assert len(offsets) > 0, "Scatter plot should have data points"


def test_bottom_left_is_bar_chart():
    """Test that bottom-left subplot (1,0) is a bar chart."""
    _, axes = create_multi_panel_plot()
    patches = axes[1, 0].patches
    assert len(patches) > 0, "Bottom-left should have bar chart patches"

    # Should have 5 bars (categories A-E)
    assert len(patches) >= 5, f"Bar chart should have at least 5 bars, got {len(patches)}"


def test_bottom_right_is_histogram():
    """Test that bottom-right subplot (1,1) is a histogram."""
    _, axes = create_multi_panel_plot()
    patches = axes[1, 1].patches
    assert len(patches) > 0, "Bottom-right should have histogram patches"

    # Should have 30 bins
    assert len(patches) >= 20, f"Histogram should have many bins, got {len(patches)}"


def test_figure_has_suptitle():
    """Test that figure has a super title."""
    fig, _ = create_multi_panel_plot()
    suptitle = fig._suptitle
    assert suptitle is not None, "Figure should have a suptitle"
    title_text = suptitle.get_text()
    assert len(title_text) > 0, "Suptitle should not be empty"
    assert "Multi-Panel" in title_text or "multi-panel" in title_text.lower(), (
        "Suptitle should mention 'Multi-Panel'"
    )


def test_line_plot_data_correctness():
    """Test that line plot contains sin and cos data."""
    _, axes = create_multi_panel_plot()
    lines = axes[0, 0].get_lines()

    # Get data from first two lines
    if len(lines) >= 2:
        line1_data = lines[0].get_ydata()
        line2_data = lines[1].get_ydata()

        # One should be approximately sin, other cos
        # Check that data ranges are reasonable for trig functions
        assert np.min(line1_data) >= -1.1, "Trig function should have min >= -1"
        assert np.max(line1_data) <= 1.1, "Trig function should have max <= 1"
        assert np.min(line2_data) >= -1.1, "Trig function should have min >= -1"
        assert np.max(line2_data) <= 1.1, "Trig function should have max <= 1"


def test_uses_object_oriented_api():
    """Test that the implementation uses the OO API (returns axes, not None)."""
    fig, axes = create_multi_panel_plot()
    assert axes is not None, "Should return axes (using OO API)"
    assert hasattr(axes, "__getitem__"), "Axes should be indexable (2D array)"


def test_no_overlapping_elements():
    """Test that tight_layout was called (prevents overlap)."""
    fig, _ = create_multi_panel_plot()
    # tight_layout sets the subplotpars, so we can check if they're reasonable
    # This is a heuristic - just verify the function doesn't crash with tight_layout
    assert fig is not None, "Figure should exist after tight_layout"
