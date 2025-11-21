"""Tests for labeled plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_labeled_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_labeled_plot = reference.create_labeled_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.array([1, 2, 3])
    y = [np.array([1, 2, 3])]
    fig = create_labeled_plot(x, y, ["Line"], "Title", "X", "Y")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_title():
    """Plot should have the specified title."""
    x = np.linspace(0, 10, 50)
    y = [np.sin(x)]
    title = "Sine Wave"
    fig = create_labeled_plot(x, y, ["sin"], title, "Time", "Amplitude")
    ax = fig.axes[0]
    assert ax.get_title() == title
    plt.close(fig)


def test_has_axis_labels():
    """Plot should have specified axis labels."""
    x = np.array([1, 2, 3])
    y = [np.array([1, 4, 9])]
    fig = create_labeled_plot(x, y, ["Square"], "Test", "Input", "Output")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Input"
    assert ax.get_ylabel() == "Output"
    plt.close(fig)


def test_has_legend():
    """Plot should have a legend."""
    x = np.array([1, 2, 3])
    y = [np.array([1, 2, 3]), np.array([3, 2, 1])]
    fig = create_labeled_plot(x, y, ["Up", "Down"], "Test", "X", "Y")
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    plt.close(fig)


def test_legend_labels():
    """Legend should have correct labels."""
    x = np.array([1, 2, 3])
    y = [np.array([1, 2, 3]), np.array([2, 4, 6])]
    labels = ["Linear", "Double"]
    fig = create_labeled_plot(x, y, labels, "Test", "X", "Y")
    ax = fig.axes[0]
    legend = ax.get_legend()
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Linear" in legend_texts
    assert "Double" in legend_texts
    plt.close(fig)


def test_legend_location():
    """Should set legend location."""
    x = np.array([1, 2, 3])
    y = [np.array([1, 2, 3])]
    fig = create_labeled_plot(x, y, ["Test"], "Title", "X", "Y", legend_loc="upper right")
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    plt.close(fig)


def test_all_elements_present():
    """Should have all labeling elements."""
    x = np.linspace(0, 2 * np.pi, 100)
    y = [np.sin(x), np.cos(x)]
    labels = ["sin(x)", "cos(x)"]
    fig = create_labeled_plot(x, y, labels, "Trigonometric Functions", "Angle (rad)", "Value")
    ax = fig.axes[0]

    assert ax.get_title() == "Trigonometric Functions"
    assert ax.get_xlabel() == "Angle (rad)"
    assert ax.get_ylabel() == "Value"
    assert ax.get_legend() is not None
    assert len(ax.lines) == 2
    plt.close(fig)
