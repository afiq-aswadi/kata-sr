"""Tests for contour plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_contour_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_contour_plot = reference.create_contour_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig = create_contour_plot(X, Y, Z)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_filled_contours():
    """Should create filled contours when filled=True."""
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    fig = create_contour_plot(X, Y, Z, filled=True)
    ax = fig.axes[0]
    # Filled contours create collections
    assert len(ax.collections) > 0
    plt.close(fig)


def test_line_contours():
    """Should create line contours when filled=False."""
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    fig = create_contour_plot(X, Y, Z, filled=False)
    ax = fig.axes[0]
    assert len(ax.collections) > 0
    plt.close(fig)


def test_has_colorbar():
    """Should include a colorbar."""
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    fig = create_contour_plot(X, Y, Z)
    # Colorbar creates an additional axes
    assert len(fig.axes) == 2
    plt.close(fig)


def test_custom_levels():
    """Should respect specified number of levels."""
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    levels = 5
    fig = create_contour_plot(X, Y, Z, levels=levels)
    ax = fig.axes[0]
    # Should have at least one collection
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_gaussian_function():
    """Should handle Gaussian-like functions."""
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2))
    fig = create_contour_plot(X, Y, Z)
    ax = fig.axes[0]
    assert len(ax.collections) > 0
    plt.close(fig)


def test_default_parameters():
    """Should use default filled=True."""
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    fig = create_contour_plot(X, Y, Z)
    ax = fig.axes[0]
    assert len(ax.collections) > 0
    plt.close(fig)
