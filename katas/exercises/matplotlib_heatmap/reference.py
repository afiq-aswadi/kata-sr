"""Reference solution for heatmap."""

import matplotlib.pyplot as plt
import numpy as np


def create_heatmap(
    data: np.ndarray,
    cmap: str = "viridis",
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a heatmap from 2D data."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
