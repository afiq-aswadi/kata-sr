"""Create plots with error bars."""

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
    """Create a plot with error bars.

    Args:
        x: X-coordinates
        y: Y-coordinates
        yerr: Y-axis error bars (optional)
        xerr: X-axis error bars (optional)
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Figure object with error bars
    """
    # BLANK_START
    raise NotImplementedError("Use plt.errorbar with yerr and xerr parameters")
    # BLANK_END
