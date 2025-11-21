"""Tests for bar chart kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_bar_chart
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_bar_chart = reference.create_bar_chart  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    categories = ["A", "B", "C"]
    values = np.array([10, 20, 15])
    fig = create_bar_chart(categories, values)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_bars():
    """Figure should contain bar chart data."""
    categories = ["X", "Y", "Z"]
    values = np.array([5, 10, 7])
    fig = create_bar_chart(categories, values)
    ax = fig.axes[0]
    # Bar chart creates Rectangle patches
    patches = ax.patches
    assert len(patches) == 3
    plt.close(fig)


def test_correct_number_of_bars():
    """Should create one bar per category."""
    categories = ["Jan", "Feb", "Mar", "Apr", "May"]
    values = np.array([10, 15, 12, 18, 20])
    fig = create_bar_chart(categories, values)
    ax = fig.axes[0]
    assert len(ax.patches) == 5
    plt.close(fig)


def test_axis_labels():
    """Axes should have the specified labels."""
    categories = ["A", "B"]
    values = np.array([1, 2])
    fig = create_bar_chart(categories, values, xlabel="Items", ylabel="Count")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Items"
    assert ax.get_ylabel() == "Count"
    plt.close(fig)


def test_default_labels():
    """Should use default labels if not specified."""
    categories = ["A", "B"]
    values = np.array([1, 2])
    fig = create_bar_chart(categories, values)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Category"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)


def test_single_bar():
    """Should handle single category."""
    categories = ["Only"]
    values = np.array([42])
    fig = create_bar_chart(categories, values)
    ax = fig.axes[0]
    assert len(ax.patches) == 1
    plt.close(fig)


def test_bar_heights():
    """Bar heights should match values."""
    categories = ["Low", "High"]
    values = np.array([5, 20])
    fig = create_bar_chart(categories, values)
    ax = fig.axes[0]
    heights = [patch.get_height() for patch in ax.patches]
    assert heights[0] == 5
    assert heights[1] == 20
    plt.close(fig)
