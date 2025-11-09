"""Add geometric shapes to highlight regions in plots."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def add_circle_highlight(
    ax: plt.Axes, center_x: float, center_y: float, radius: float
) -> None:
    """Add a circular highlight to emphasize a region.

    The circle should have:
    - center at (center_x, center_y)
    - given radius
    - facecolor='none' (transparent fill)
    - edgecolor='green'
    - linewidth=2
    - alpha=0.6

    Args:
        ax: matplotlib axes object
        center_x: x coordinate of circle center
        center_y: y coordinate of circle center
        radius: circle radius
    """
    # TODO: Create a Circle patch with specified center and radius
    # TODO: Set facecolor, edgecolor, linewidth, and alpha
    # TODO: Add the patch to the axes with ax.add_patch()
    # BLANK_START
    raise NotImplementedError("Create Circle patch and add to axes")
    # BLANK_END


def add_rectangle_highlight(
    ax: plt.Axes, x_start: float, y_start: float, width: float, height: float
) -> None:
    """Add a rectangular highlight to emphasize a region.

    The rectangle should have:
    - bottom-left corner at (x_start, y_start)
    - given width and height
    - facecolor='blue'
    - alpha=0.2
    - edgecolor='navy'
    - linewidth=1.5

    Args:
        ax: matplotlib axes object
        x_start: x coordinate of bottom-left corner
        y_start: y coordinate of bottom-left corner
        width: rectangle width
        height: rectangle height
    """
    # TODO: Create a Rectangle patch with specified position and dimensions
    # TODO: Set facecolor, alpha, edgecolor, and linewidth
    # TODO: Add the patch to the axes with ax.add_patch()
    # BLANK_START
    raise NotImplementedError("Create Rectangle patch and add to axes")
    # BLANK_END
