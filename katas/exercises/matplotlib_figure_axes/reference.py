"""Reference implementation for matplotlib figure/axes kata."""

import matplotlib.pyplot as plt
import numpy as np


def create_line_plot():
    """Create a styled line plot using matplotlib's object-oriented interface.

    Demonstrates the fundamental fig, ax = plt.subplots() pattern for creating
    publication-quality plots with proper labels, title, legend, and grid.

    Returns:
        tuple: (fig, ax) where fig is the Figure object and ax is the Axes object
    """
    # BLANK_START
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Plot two lines
    ax.plot(x, y_sin, "r-", label="sin(x)", linewidth=2)
    ax.plot(x, y_cos, "b--", label="cos(x)", linewidth=2)

    # Customize the plot
    ax.set_title("Trigonometric Functions", fontsize=14, fontweight="bold")
    ax.set_xlabel("x (radians)", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig, ax
    # BLANK_END
