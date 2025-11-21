"""Tests for multi-line plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_multi_line_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_multi_line_plot = reference.create_multi_line_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.array([1, 2, 3])
    y1 = np.array([1, 2, 3])
    y2 = np.array([3, 2, 1])
    fig = create_multi_line_plot(x, [y1, y2], ["Line 1", "Line 2"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_correct_number_of_lines():
    """Should plot all specified lines."""
    x = np.linspace(0, 10, 50)
    y_series = [np.sin(x), np.cos(x), x/5]
    labels = ["sin", "cos", "linear"]
    fig = create_multi_line_plot(x, y_series, labels)
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    plt.close(fig)


def test_has_legend():
    """Should include a legend."""
    x = np.array([1, 2, 3])
    y1 = np.array([1, 2, 3])
    y2 = np.array([3, 2, 1])
    fig = create_multi_line_plot(x, [y1, y2], ["A", "B"])
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    plt.close(fig)


def test_legend_labels():
    """Legend should have correct labels."""
    x = np.array([1, 2, 3])
    y_series = [np.array([1, 2, 3]), np.array([3, 2, 1])]
    labels = ["First", "Second"]
    fig = create_multi_line_plot(x, y_series, labels)
    ax = fig.axes[0]
    legend = ax.get_legend()
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "First" in legend_texts
    assert "Second" in legend_texts
    plt.close(fig)


def test_linestyles():
    """Should apply specified linestyles."""
    x = np.array([1, 2, 3])
    y_series = [np.array([1, 2, 3]), np.array([3, 2, 1])]
    labels = ["A", "B"]
    linestyles = ['-', '--']
    fig = create_multi_line_plot(x, y_series, labels, linestyles)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.lines[0].get_linestyle() == '-'
    assert ax.lines[1].get_linestyle() == '--'
    plt.close(fig)


def test_single_line():
    """Should handle single line."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    fig = create_multi_line_plot(x, [y], ["Linear"])
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    plt.close(fig)


def test_default_linestyle():
    """Should use solid lines when linestyles not specified."""
    x = np.array([1, 2, 3])
    y_series = [np.array([1, 2, 3]), np.array([3, 2, 1])]
    labels = ["A", "B"]
    fig = create_multi_line_plot(x, y_series, labels)
    ax = fig.axes[0]
    # Default should be solid line
    assert ax.lines[0].get_linestyle() == '-'
    plt.close(fig)
