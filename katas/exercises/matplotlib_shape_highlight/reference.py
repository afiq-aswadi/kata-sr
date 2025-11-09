"""Add geometric shapes to highlight regions in plots - reference solution."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def add_circle_highlight(
    ax: plt.Axes, center_x: float, center_y: float, radius: float
) -> None:
    """Add a circular highlight to emphasize a region."""
    circle = Circle(
        (center_x, center_y),
        radius,
        facecolor="none",
        edgecolor="green",
        linewidth=2,
        alpha=0.6,
    )
    ax.add_patch(circle)


def add_rectangle_highlight(
    ax: plt.Axes, x_start: float, y_start: float, width: float, height: float
) -> None:
    """Add a rectangular highlight to emphasize a region."""
    rect = Rectangle(
        (x_start, y_start),
        width,
        height,
        facecolor="blue",
        alpha=0.2,
        edgecolor="navy",
        linewidth=1.5,
    )
    ax.add_patch(rect)
