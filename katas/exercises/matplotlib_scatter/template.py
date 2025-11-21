"""Create a basic scatter plot."""

import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a scatter plot with labeled axes.

    Args:
        x: X-coordinates of points
        y: Y-coordinates of points
        xlabel: Label for x-axis
        ylabel: Label for y-axis

    Returns:
        Figure object containing the scatter plot
    """
    # BLANK_START
    raise NotImplementedError("Create scatter plot with plt.scatter and add labels")
    # BLANK_END
