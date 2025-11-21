"""Tests for histogram kata."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from user_kata import create_histogram
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    create_histogram = reference.create_histogram  # type: ignore


def test_returns_figure():
    """Should return a Figure object."""
    data = np.random.randn(100)
    fig = create_histogram(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_has_histogram_patches():
    """Figure should contain histogram bars."""
    data = np.random.randn(100)
    fig = create_histogram(data, bins=5)
    ax = fig.axes[0]
    # Histogram creates Rectangle patches
    assert len(ax.patches) == 5
    plt.close(fig)


def test_correct_number_of_bins():
    """Should create specified number of bins."""
    data = np.linspace(0, 10, 100)
    for bins in [5, 10, 20]:
        fig = create_histogram(data, bins=bins)
        ax = fig.axes[0]
        assert len(ax.patches) == bins
        plt.close(fig)


def test_alpha_transparency():
    """Bars should have specified alpha value."""
    data = np.random.randn(100)
    alpha_value = 0.5
    fig = create_histogram(data, alpha=alpha_value)
    ax = fig.axes[0]
    # Check alpha of first patch
    assert ax.patches[0].get_alpha() == alpha_value
    plt.close(fig)


def test_axis_labels():
    """Axes should have the specified labels."""
    data = np.random.randn(50)
    fig = create_histogram(data, xlabel="Score", ylabel="Count")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Score"
    assert ax.get_ylabel() == "Count"
    plt.close(fig)


def test_default_labels():
    """Should use default labels if not specified."""
    data = np.random.randn(50)
    fig = create_histogram(data)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Value"
    assert ax.get_ylabel() == "Frequency"
    plt.close(fig)


def test_default_parameters():
    """Should use default bins and alpha."""
    data = np.random.randn(100)
    fig = create_histogram(data)
    ax = fig.axes[0]
    assert len(ax.patches) == 10  # default bins
    assert ax.patches[0].get_alpha() == 0.7  # default alpha
    plt.close(fig)
