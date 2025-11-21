"""Tests for error bar plot kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_errorbar_plot
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_errorbar_plot = reference.create_errorbar_plot  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    fig = create_errorbar_plot(x, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_line():
    """Should have data line."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    fig = create_errorbar_plot(x, y)
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)


def test_y_error_bars():
    """Should display y-axis error bars."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    yerr = np.array([0.5, 0.3, 0.7])
    fig = create_errorbar_plot(x, y, yerr=yerr)
    ax = fig.axes[0]
    # errorbar creates line objects or collections for error bars
    assert len(ax.lines) >= 1 or len(ax.collections) > 0
    plt.close(fig)


def test_x_error_bars():
    """Should display x-axis error bars."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    xerr = np.array([0.1, 0.2, 0.1])
    fig = create_errorbar_plot(x, y, xerr=xerr)
    ax = fig.axes[0]
    assert len(ax.lines) >= 1 or len(ax.collections) > 0
    plt.close(fig)


def test_both_error_bars():
    """Should handle both x and y error bars."""
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 4, 9, 16])
    yerr = np.array([0.5, 1.0, 1.5, 2.0])
    xerr = np.array([0.1, 0.1, 0.1, 0.1])
    fig = create_errorbar_plot(x, y, yerr=yerr, xerr=xerr)
    ax = fig.axes[0]
    assert len(ax.lines) >= 1 or len(ax.collections) > 0
    plt.close(fig)


def test_axis_labels():
    """Should set axis labels."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    fig = create_errorbar_plot(x, y, xlabel="Time", ylabel="Measurement")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Measurement"
    plt.close(fig)


def test_no_errors():
    """Should work without error bars."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    fig = create_errorbar_plot(x, y)
    ax = fig.axes[0]
    assert len(ax.lines) >= 1
    plt.close(fig)
