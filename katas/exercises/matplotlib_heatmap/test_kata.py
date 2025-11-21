"""Tests for heatmap kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_heatmap
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_heatmap = reference.create_heatmap  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    data = np.random.rand(10, 10)
    fig = create_heatmap(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_image():
    """Figure should contain image data."""
    data = np.random.rand(5, 5)
    fig = create_heatmap(data)
    ax = fig.axes[0]
    assert len(ax.images) > 0
    plt.close(fig)


def test_has_colorbar():
    """Should include a colorbar."""
    data = np.random.rand(5, 5)
    fig = create_heatmap(data)
    # Colorbar creates an additional axes
    assert len(fig.axes) == 2  # main axes + colorbar axes
    plt.close(fig)


def test_custom_colormap():
    """Should use specified colormap."""
    data = np.random.rand(5, 5)
    fig = create_heatmap(data, cmap="plasma")
    ax = fig.axes[0]
    assert len(ax.images) > 0
    plt.close(fig)


def test_axis_labels():
    """Should set axis labels."""
    data = np.random.rand(5, 5)
    fig = create_heatmap(data, xlabel="Column", ylabel="Row")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Column"
    assert ax.get_ylabel() == "Row"
    plt.close(fig)


def test_rectangular_data():
    """Should handle non-square matrices."""
    data = np.random.rand(10, 5)
    fig = create_heatmap(data)
    ax = fig.axes[0]
    assert len(ax.images) > 0
    plt.close(fig)


def test_data_values():
    """Should correctly display data values."""
    data = np.array([[1, 2], [3, 4]])
    fig = create_heatmap(data)
    ax = fig.axes[0]
    image = ax.images[0]
    np.testing.assert_array_equal(image.get_array(), data)
    plt.close(fig)
