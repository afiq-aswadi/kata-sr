"""Create plots with dual y-axes for different scales."""

import matplotlib.pyplot as plt
import numpy as np


def create_dual_axis_plot(
    x: np.ndarray, y1: np.ndarray, y2: np.ndarray
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a plot with dual y-axes showing data at different scales.

    The left axis (ax1) should plot y1 in blue with label "Primary (blue)".
    The right axis (ax2) should plot y2 in red with label "Secondary (red)".
    Both axes should have matching tick label colors.

    Args:
        x: x-axis data
        y1: left y-axis data (primary)
        y2: right y-axis data (secondary)

    Returns:
        tuple: (fig, ax1, ax2) where ax1 is left axis, ax2 is right axis
    """
    # TODO: Create figure and primary axis
    # TODO: Plot y1 on primary axis with color 'tab:blue'
    # TODO: Set ylabel and tick colors for primary axis
    # TODO: Create secondary axis using twinx()
    # TODO: Plot y2 on secondary axis with color 'tab:red'
    # TODO: Set ylabel and tick colors for secondary axis
    # BLANK_START
    raise NotImplementedError("Create dual axes with twinx() and color-coded labels")
    # BLANK_END
