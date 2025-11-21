"""Reference solution for histogram."""

import matplotlib.pyplot as plt
import numpy as np


def create_histogram(
    data: np.ndarray,
    bins: int = 10,
    alpha: float = 0.7,
    xlabel: str = "Value",
    ylabel: str = "Frequency"
) -> plt.Figure:
    """Create a histogram with customizable bins and transparency."""
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
