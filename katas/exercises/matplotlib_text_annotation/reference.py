"""Add styled text annotations to matplotlib plots - reference solution."""

import matplotlib.pyplot as plt


def add_text_annotation(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Add a text annotation with custom styling."""
    ax.text(
        x,
        y,
        text,
        fontsize=12,
        bbox=dict(facecolor="yellow", alpha=0.5),
        horizontalalignment="center",
        verticalalignment="bottom",
    )
