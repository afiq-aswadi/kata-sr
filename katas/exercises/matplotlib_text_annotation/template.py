"""Add styled text annotations to matplotlib plots."""

import matplotlib.pyplot as plt


def add_text_annotation(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Add a text annotation with custom styling.

    The text should have:
    - fontsize=12
    - bbox with facecolor='yellow' and alpha=0.5
    - horizontalalignment='center'
    - verticalalignment='bottom'

    Args:
        ax: matplotlib axes object
        x: x coordinate in data coordinates
        y: y coordinate in data coordinates
        text: annotation text
    """
    # TODO: Use ax.text() to add annotation at (x, y)
    # TODO: Include bbox parameter as dict with facecolor and alpha
    # TODO: Set alignment parameters
    # BLANK_START
    raise NotImplementedError("Add text with bbox styling and alignment")
    # BLANK_END
