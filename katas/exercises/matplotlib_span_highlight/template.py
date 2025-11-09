"""Add vertical and horizontal spans to highlight axis regions."""

import matplotlib.pyplot as plt


def add_vertical_span(
    ax: plt.Axes, x_start: float, x_end: float, color: str = "gray"
) -> None:
    """Add a vertical span to highlight a region on the x-axis.

    The span should:
    - Extend from x_start to x_end
    - Use the given color
    - Have alpha=0.3 for transparency
    - Have zorder=0 to place behind the data

    Args:
        ax: matplotlib axes object
        x_start: start x coordinate
        x_end: end x coordinate
        color: span color
    """
    # TODO: Use ax.axvspan() to add a vertical span
    # TODO: Set alpha=0.3 and zorder=0
    # BLANK_START
    raise NotImplementedError("Add vertical span with axvspan()")
    # BLANK_END


def add_horizontal_span(
    ax: plt.Axes, y_start: float, y_end: float, color: str = "gray"
) -> None:
    """Add a horizontal span to highlight a region on the y-axis.

    The span should:
    - Extend from y_start to y_end
    - Use the given color
    - Have alpha=0.3 for transparency
    - Have zorder=0 to place behind the data

    Args:
        ax: matplotlib axes object
        y_start: start y coordinate
        y_end: end y coordinate
        color: span color
    """
    # TODO: Use ax.axhspan() to add a horizontal span
    # TODO: Set alpha=0.3 and zorder=0
    # BLANK_START
    raise NotImplementedError("Add horizontal span with axhspan()")
    # BLANK_END
