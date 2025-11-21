"""Tests for box plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_box_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_box_plot = reference.create_box_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    data = [np.random.randn(50)]
    fig = create_box_plot(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_single_box():
    """Should handle single dataset."""
    data = [np.array([1, 2, 3, 4, 5])]
    fig = create_box_plot(data)
    ax = fig.axes[0]
    # Box plot creates line objects for boxes, whiskers, etc.
    assert len(ax.lines) > 0
    plt.close(fig)


def test_multiple_boxes():
    """Should create multiple boxes."""
    data = [
        np.random.randn(50),
        np.random.randn(50) + 1,
        np.random.randn(50) - 1
    ]
    fig = create_box_plot(data)
    ax = fig.axes[0]
    # Multiple datasets create more line objects
    assert len(ax.lines) > 0
    plt.close(fig)


def test_labels():
    """Should set tick labels for datasets."""
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    labels = ["Group A", "Group B"]
    fig = create_box_plot(data, labels=labels)
    ax = fig.axes[0]
    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert "Group A" in tick_labels
    assert "Group B" in tick_labels
    plt.close(fig)


def test_axis_labels():
    """Should set axis labels."""
    data = [np.random.randn(20)]
    fig = create_box_plot(data, xlabel="Category", ylabel="Score")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Category"
    assert ax.get_ylabel() == "Score"
    plt.close(fig)


def test_default_labels():
    """Should use default labels."""
    data = [np.random.randn(20)]
    fig = create_box_plot(data)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Dataset"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)


def test_different_sizes():
    """Should handle datasets of different sizes."""
    data = [
        np.random.randn(10),
        np.random.randn(50),
        np.random.randn(30)
    ]
    fig = create_box_plot(data)
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)
