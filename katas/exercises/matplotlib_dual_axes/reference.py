"""Create plots with dual y-axes for different scales - reference solution."""

import matplotlib.pyplot as plt
import numpy as np


def create_dual_axis_plot(
    x: np.ndarray, y1: np.ndarray, y2: np.ndarray
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a plot with dual y-axes showing data at different scales."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis (left)
    ax1.plot(x, y1, color="tab:blue", label="Primary")
    ax1.set_ylabel("Primary (blue)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Secondary axis (right)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="tab:red", label="Secondary")
    ax2.set_ylabel("Secondary (red)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    return fig, ax1, ax2
