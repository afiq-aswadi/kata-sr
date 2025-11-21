"""Reference solution for scatter plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a scatter plot with labeled axes."""
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
