"""Create line plots with multiple series."""

import matplotlib.pyplot as plt
import numpy as np


def create_multi_line_plot(
    x: np.ndarray,
    y_series: list[np.ndarray],
    labels: list[str],
    linestyles: list[str] | None = None
) -> plt.Figure:
    """Plot multiple lines on the same axes.

    Args:
        x: X-coordinates (shared across all lines)
        y_series: List of y-coordinate arrays for each line
        labels: List of labels for each line (for legend)
        linestyles: Optional list of linestyles (e.g., ['-', '--', ':'])

    Returns:
        Figure object containing the line plots
    """
    # BLANK_START
    raise NotImplementedError("Plot each line with plt.plot, use linestyles, add legend")
    # BLANK_END
