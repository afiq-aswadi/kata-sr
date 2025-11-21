"""Customize colors and markers in plots."""

import matplotlib.pyplot as plt
import numpy as np


def create_styled_scatter(
    x: np.ndarray,
    y: np.ndarray,
    color: str = "blue",
    marker: str = "o",
    size: float | np.ndarray = 50,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> plt.Figure:
    """Create a scatter plot with custom styling.

    Args:
        x: X-coordinates
        y: Y-coordinates
        color: Color (name like 'red' or hex like '#FF0000')
        marker: Marker style ('o', 's', '^', 'D', etc.)
        size: Marker size (scalar) or array of sizes
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Figure object with styled scatter plot
    """
    # BLANK_START
    raise NotImplementedError("Use plt.scatter with color, marker, and s (size) parameters")
    # BLANK_END
