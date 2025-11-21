"""Reference solution for styled scatter plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_styled_scatter(
    x: np.ndarray,
    y: np.ndarray,
    color: str = "blue",
    marker: str = "o",
    size: float | np.ndarray = 50,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a scatter plot with custom styling."""
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=color, marker=marker, s=size)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
