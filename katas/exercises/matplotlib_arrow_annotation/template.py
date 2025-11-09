"""Add arrow annotations to matplotlib plots."""

import matplotlib.pyplot as plt


def add_arrow_annotation(
    ax: plt.Axes,
    x_point: float,
    y_point: float,
    text: str,
    offset_x: float = 20,
    offset_y: float = 20,
) -> None:
    """Add an arrow annotation pointing to a specific data point.

    The annotation should:
    - Point to (x_point, y_point) in data coordinates
    - Have text offset by (offset_x, offset_y) in points
    - Use arrowprops with arrowstyle='->', color='red', lw=2
    - Have bbox with boxstyle='round', facecolor='wheat', alpha=0.5
    - Use fontsize=10

    Args:
        ax: matplotlib axes object
        x_point: x coordinate of the point to annotate
        y_point: y coordinate of the point to annotate
        text: annotation text
        offset_x: x offset in points for text position
        offset_y: y offset in points for text position
    """
    # TODO: Use ax.annotate() with xy=(x_point, y_point)
    # TODO: Set xytext=(offset_x, offset_y) with textcoords='offset points'
    # TODO: Create arrowprops dict with arrowstyle, color, and linewidth
    # TODO: Create bbox dict with boxstyle, facecolor, and alpha
    # BLANK_START
    raise NotImplementedError("Add arrow annotation with annotate()")
    # BLANK_END
