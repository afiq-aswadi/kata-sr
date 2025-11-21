"""Add labels, titles, and legends to plots."""

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
    """Create a plot with complete labeling.

    Args:
        x: X-coordinates (shared across all lines)
        y_series: List of y-coordinate arrays for each line
        labels: List of labels for legend
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend_loc: Legend location ('best', 'upper right', etc.)

    Returns:
        Figure object with title, axis labels, and legend
    """
    # BLANK_START
    raise NotImplementedError("Plot lines, add title with plt.title, labels, and legend with loc")
    # BLANK_END
