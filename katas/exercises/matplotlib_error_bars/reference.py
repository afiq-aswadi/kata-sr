"""Reference solution for error bar plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_errorbar_plot(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    xerr: np.ndarray | None = None,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a plot with error bars."""
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, xerr=xerr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
