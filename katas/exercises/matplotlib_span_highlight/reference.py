"""Add vertical and horizontal spans to highlight axis regions - reference solution."""

import matplotlib.pyplot as plt


def add_vertical_span(
    ax: plt.Axes, x_start: float, x_end: float, color: str = "gray"
) -> None:
    """Add a vertical span to highlight a region on the x-axis."""
    ax.axvspan(x_start, x_end, alpha=0.3, color=color, zorder=0)


def add_horizontal_span(
    ax: plt.Axes, y_start: float, y_end: float, color: str = "gray"
) -> None:
    """Add a horizontal span to highlight a region on the y-axis."""
    ax.axhspan(y_start, y_end, alpha=0.3, color=color, zorder=0)
