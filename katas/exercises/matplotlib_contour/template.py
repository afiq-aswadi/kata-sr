"""Create contour plots."""

import matplotlib.pyplot as plt
import numpy as np


def create_contour_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    filled: bool = True,
    levels: int = 10
) -> plt.Figure:
    """Create a contour plot.

    Args:
        X: X-coordinate grid (from np.meshgrid)
        Y: Y-coordinate grid (from np.meshgrid)
        Z: Function values at each grid point
        filled: If True, use filled contours (contourf), else lines (contour)
        levels: Number of contour levels

    Returns:
        Figure object with contour plot and colorbar
    """
    # BLANK_START
    raise NotImplementedError("Use plt.contourf or plt.contour, then plt.colorbar()")
    # BLANK_END
