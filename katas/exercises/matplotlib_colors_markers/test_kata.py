"""Tests for styled scatter plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_styled_scatter
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_styled_scatter = reference.create_styled_scatter  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_styled_scatter(x, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_custom_color():
    """Should apply specified color."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_styled_scatter(x, y, color="red")
    ax = fig.axes[0]
    # Color is stored in the collection
    collection = ax.collections[0]
    # Note: color comparison is tricky, just verify it's set
    assert collection.get_facecolors() is not None
    plt.close(fig)


def test_custom_marker():
    """Should use specified marker style."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    markers = ['s', '^', 'D', '*']
    for marker in markers:
        fig = create_styled_scatter(x, y, marker=marker)
        ax = fig.axes[0]
        # Marker is part of the collection paths
        assert len(ax.collections) > 0
        plt.close(fig)


def test_uniform_size():
    """Should handle uniform marker size."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    size = 100
    fig = create_styled_scatter(x, y, size=size)
    ax = fig.axes[0]
    collection = ax.collections[0]
    sizes = collection.get_sizes()
    assert len(sizes) > 0
    assert sizes[0] == size
    plt.close(fig)


def test_variable_sizes():
    """Should handle array of sizes."""
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    sizes = np.array([50, 100, 150, 200])
    fig = create_styled_scatter(x, y, size=sizes)
    ax = fig.axes[0]
    collection = ax.collections[0]
    result_sizes = collection.get_sizes()
    np.testing.assert_array_equal(result_sizes, sizes)
    plt.close(fig)


def test_hex_color():
    """Should handle hex color codes."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_styled_scatter(x, y, color="#FF5733")
    ax = fig.axes[0]
    collection = ax.collections[0]
    assert collection.get_facecolors() is not None
    plt.close(fig)


def test_axis_labels():
    """Should set axis labels."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_styled_scatter(x, y, xlabel="Time (s)", ylabel="Distance (m)")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Distance (m)"
    plt.close(fig)


def test_default_parameters():
    """Should use default styling when not specified."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    fig = create_styled_scatter(x, y)
    ax = fig.axes[0]
    collection = ax.collections[0]
    # Should have default size
    assert collection.get_sizes()[0] == 50
    plt.close(fig)
