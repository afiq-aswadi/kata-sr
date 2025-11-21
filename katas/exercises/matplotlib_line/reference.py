"""Reference solution for multi-line plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_multi_line_plot(
    x: np.ndarray,
    y_series: list[np.ndarray],
    labels: list[str],
    linestyles: list[str] | None = None
) -> plt.Figure:
    """Plot multiple lines on the same axes."""
    fig, ax = plt.subplots()

    for i, y in enumerate(y_series):
        linestyle = linestyles[i] if linestyles else '-'
        ax.plot(x, y, label=labels[i], linestyle=linestyle)

    ax.legend()
    return fig
