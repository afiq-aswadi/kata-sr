"""Reference solution for bar chart."""

import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart(
    categories: list[str],
    values: np.ndarray,
    xlabel: str = "Category",
    ylabel: str = "Value"
) -> plt.Figure:
    """Create a vertical bar chart."""
    fig, ax = plt.subplots()
    ax.bar(categories, values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
