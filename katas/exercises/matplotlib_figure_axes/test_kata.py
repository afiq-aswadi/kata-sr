"""Tests for matplotlib figure/axes kata."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-GUI backend for testing

try:
    from user_kata import create_line_plot
except ModuleNotFoundError:  # pragma: no cover - fallback for standalone test runs
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)

    create_line_plot = reference.create_line_plot  # type: ignore[attr-defined]


def test_returns_figure_and_axes():
    """Test that function returns a figure and axes."""
    result = create_line_plot()
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Should return (fig, ax) tuple"
    fig, ax = result
    assert hasattr(fig, "get_size_inches"), "First element should be a Figure"
    assert hasattr(ax, "set_title"), "Second element should be an Axes"


def test_figure_size():
    """Test that figure has correct size (10x6)."""
    fig, _ = create_line_plot()
    width, height = fig.get_size_inches()
    assert abs(width - 10) < 0.1, f"Figure width should be 10, got {width}"
    assert abs(height - 6) < 0.1, f"Figure height should be 6, got {height}"


def test_has_title():
    """Test that axes has a title."""
    _, ax = create_line_plot()
    title = ax.get_title()
    assert title and len(title) > 0, "Plot should have a title"
    assert "Trigonometric" in title or "trig" in title.lower(), (
        "Title should mention trigonometric functions"
    )


def test_has_labels():
    """Test that axes has x and y labels."""
    _, ax = create_line_plot()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    assert xlabel and len(xlabel) > 0, "Plot should have x label"
    assert ylabel and len(ylabel) > 0, "Plot should have y label"
    assert "radian" in xlabel.lower() or "x" in xlabel.lower(), (
        "X label should mention x or radians"
    )


def test_has_grid():
    """Test that grid is enabled."""
    _, ax = create_line_plot()
    gridlines_x = ax.xaxis.get_gridlines()
    gridlines_y = ax.yaxis.get_gridlines()
    has_grid = (len(gridlines_x) > 0 and gridlines_x[0].get_visible()) or (
        len(gridlines_y) > 0 and gridlines_y[0].get_visible()
    )
    assert has_grid, "Plot should have grid enabled"


def test_has_two_lines():
    """Test that plot contains exactly 2 lines (sin and cos)."""
    _, ax = create_line_plot()
    lines = ax.get_lines()
    assert len(lines) >= 2, f"Plot should have at least 2 lines, got {len(lines)}"


def test_line_colors():
    """Test that lines have red and blue colors."""
    _, ax = create_line_plot()
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]

    # Check for red
    has_red = any(c in ["r", "red", (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)] for c in colors)
    # Check for blue
    has_blue = any(c in ["b", "blue", (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0)] for c in colors)

    assert has_red, "Plot should have a red line for sin(x)"
    assert has_blue, "Plot should have a blue line for cos(x)"


def test_line_styles():
    """Test that lines have different styles (solid and dashed)."""
    _, ax = create_line_plot()
    lines = ax.get_lines()
    linestyles = [line.get_linestyle() for line in lines]

    # Should have both solid and dashed
    has_solid = any(ls in ["-", "solid"] for ls in linestyles)
    has_dashed = any(ls in ["--", "dashed"] for ls in linestyles)

    assert has_solid, "Plot should have a solid line"
    assert has_dashed, "Plot should have a dashed line"


def test_has_legend():
    """Test that axes has a legend."""
    _, ax = create_line_plot()
    legend = ax.get_legend()
    assert legend is not None, "Plot should have a legend"
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) >= 2, "Legend should have at least 2 entries"


def test_legend_labels():
    """Test that legend contains sin and cos labels."""
    _, ax = create_line_plot()
    legend = ax.get_legend()
    legend_texts = [t.get_text().lower() for t in legend.get_texts()]

    has_sin = any("sin" in text for text in legend_texts)
    has_cos = any("cos" in text for text in legend_texts)

    assert has_sin, "Legend should mention sin(x)"
    assert has_cos, "Legend should mention cos(x)"


def test_data_range():
    """Test that plot data covers appropriate range for trig functions."""
    _, ax = create_line_plot()
    lines = ax.get_lines()

    for line in lines[:2]:  # Check first two lines
        ydata = line.get_ydata()
        assert np.min(ydata) >= -1.1, "Trig function should have min >= -1"
        assert np.max(ydata) <= 1.1, "Trig function should have max <= 1"


def test_uses_oo_api():
    """Test that implementation uses OO API (returns ax, not None)."""
    fig, ax = create_line_plot()
    assert ax is not None, "Should return axes (using OO API)"
    assert hasattr(ax, "plot"), "Axes should have plot method"


def test_x_data_range():
    """Test that x data spans approximately 0 to 2π."""
    _, ax = create_line_plot()
    lines = ax.get_lines()
    if len(lines) > 0:
        xdata = lines[0].get_xdata()
        assert np.min(xdata) < 0.5, "X data should start near 0"
        assert np.max(xdata) > 6.0, "X data should go to approximately 2π (6.28)"


def test_sufficient_data_points():
    """Test that lines have enough data points for smooth curves."""
    _, ax = create_line_plot()
    lines = ax.get_lines()
    if len(lines) > 0:
        xdata = lines[0].get_xdata()
        assert len(xdata) >= 50, f"Should have at least 50 data points, got {len(xdata)}"
