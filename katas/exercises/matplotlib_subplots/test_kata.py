"""Tests for subplots kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_subplot_grid
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_subplot_grid = reference.create_subplot_grid  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    plot_data = [{
        'row': 0, 'col': 0, 'type': 'scatter',
        'data': {'x': np.array([1, 2, 3]), 'y': np.array([1, 2, 3])}
    }]
    fig = create_subplot_grid(1, 1, plot_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_correct_number_of_subplots():
    """Should create correct grid dimensions."""
    plot_data = []
    fig = create_subplot_grid(2, 2, plot_data)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_scatter_subplot():
    """Should create scatter plot in specified position."""
    plot_data = [{
        'row': 0, 'col': 0, 'type': 'scatter',
        'data': {'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])}
    }]
    fig = create_subplot_grid(1, 1, plot_data)
    ax = fig.axes[0]
    assert len(ax.collections) > 0
    plt.close(fig)


def test_line_subplot():
    """Should create line plot in specified position."""
    plot_data = [{
        'row': 0, 'col': 0, 'type': 'line',
        'data': {'x': np.array([1, 2, 3]), 'y': np.array([1, 4, 9])}
    }]
    fig = create_subplot_grid(1, 1, plot_data)
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)


def test_bar_subplot():
    """Should create bar plot in specified position."""
    plot_data = [{
        'row': 0, 'col': 0, 'type': 'bar',
        'data': {'categories': ['A', 'B'], 'values': np.array([1, 2])}
    }]
    fig = create_subplot_grid(1, 1, plot_data)
    ax = fig.axes[0]
    assert len(ax.patches) > 0
    plt.close(fig)


def test_hist_subplot():
    """Should create histogram in specified position."""
    plot_data = [{
        'row': 0, 'col': 0, 'type': 'hist',
        'data': {'values': np.random.randn(100), 'bins': 10}
    }]
    fig = create_subplot_grid(1, 1, plot_data)
    ax = fig.axes[0]
    assert len(ax.patches) > 0
    plt.close(fig)


def test_multiple_subplots():
    """Should handle multiple plots in different positions."""
    plot_data = [
        {'row': 0, 'col': 0, 'type': 'scatter',
         'data': {'x': np.array([1, 2]), 'y': np.array([1, 2])}},
        {'row': 0, 'col': 1, 'type': 'line',
         'data': {'x': np.array([1, 2]), 'y': np.array([2, 4])}},
    ]
    fig = create_subplot_grid(1, 2, plot_data)
    assert len(fig.axes) == 2
    plt.close(fig)
