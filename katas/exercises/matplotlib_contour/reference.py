"""Reference solution for contour plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_contour_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filled: bool = True,
    levels: int = 10
) -> plt.Figure:
    """Create a contour plot."""
    fig, ax = plt.subplots()
    if filled:
        cs = ax.contourf(X, Y, Z, levels=levels)
    else:
        cs = ax.contour(X, Y, Z, levels=levels)
    plt.colorbar(cs, ax=ax)
    return fig
