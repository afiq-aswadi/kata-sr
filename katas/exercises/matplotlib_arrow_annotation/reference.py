"""Add arrow annotations to matplotlib plots - reference solution."""

import matplotlib.pyplot as plt


def add_arrow_annotation(
    ax: plt.Axes,
    x_point: float,
    y_point: float,
    text: str,
    offset_x: float = 20,
    offset_y: float = 20,
) -> None:
    """Add an arrow annotation pointing to a specific data point."""
    ax.annotate(
        text,
        xy=(x_point, y_point),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )
