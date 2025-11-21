"""Reference solution for labeled plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_labeled_plot(
    x: np.ndarray,
    y_series: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    legend_loc: str = "best"
) -> plt.Figure:
    """Create a plot with complete labeling."""
    fig, ax = plt.subplots()

    for y, label in zip(y_series, labels):
        ax.plot(x, y, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    return fig
