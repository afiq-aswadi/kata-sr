"""Tests for scatter plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_scatter_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_scatter_plot = reference.create_scatter_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_scatter_plot(x, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_scatter_collection():
    """Figure should contain scatter plot data."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_scatter_plot(x, y)
    ax = fig.axes[0]
    # Check for PathCollection (scatter creates this)
    assert len(ax.collections) > 0
    plt.close(fig)


def test_correct_number_of_points():
    """All data points should be plotted."""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    fig = create_scatter_plot(x, y)
    ax = fig.axes[0]
    # Get scatter collection
    collection = ax.collections[0]
    assert len(collection.get_offsets()) == 20
    plt.close(fig)


def test_axis_labels():
    """Axes should have the specified labels."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_scatter_plot(x, y, xlabel="Time", ylabel="Value")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)


def test_default_labels():
    """Should use default labels if not specified."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_scatter_plot(x, y)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)


def test_single_point():
    """Should handle single data point."""
    x = np.array([5.0])
    y = np.array([10.0])
    fig = create_scatter_plot(x, y)
    ax = fig.axes[0]
    collection = ax.collections[0]
    assert len(collection.get_offsets()) == 1
    plt.close(fig)


def test_large_dataset():
    """Should handle large datasets."""
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    fig = create_scatter_plot(x, y)
    ax = fig.axes[0]
    collection = ax.collections[0]
    assert len(collection.get_offsets()) == 1000
    plt.close(fig)
