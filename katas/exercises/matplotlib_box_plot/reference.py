"""Reference solution for box plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_box_plot(
    datasets: list[np.ndarray],
    labels: list[str] | None = None,
    xlabel: str = "Dataset",
    ylabel: str = "Value"
) -> plt.Figure:
    """Create a box plot for multiple datasets."""
    fig, ax = plt.subplots()
    ax.boxplot(datasets, labels=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
