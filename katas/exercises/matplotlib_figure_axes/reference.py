"""Reference implementation for matplotlib figure/axes kata."""

import matplotlib.pyplot as plt
import numpy as np


def create_multi_panel_plot():
    """Create a 2x2 multi-panel plot demonstrating the figure/axes pattern.

    This function creates a figure with four different plot types:
    - Top-left: Line plot with sin wave
    - Top-right: Scatter plot with random data
    - Bottom-left: Bar chart
    - Bottom-right: Histogram

    Each subplot has proper titles, labels, legends, and styling.

    Returns:
        tuple: (fig, axes) where fig is the Figure object and axes is a 2x2 ndarray of Axes
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Top-left: Line plot
    axes[0, 0].plot(x, y_sin, "r-", label="sin(x)", linewidth=2)
    axes[0, 0].plot(x, y_cos, "b--", label="cos(x)", linewidth=2)
    axes[0, 0].set_title("Line Plot: Trigonometric Functions", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("x (radians)")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Scatter plot
    np.random.seed(42)
    scatter_x = np.random.randn(100)
    scatter_y = np.random.randn(100)
    axes[0, 1].scatter(scatter_x, scatter_y, c="blue", marker="o", alpha=0.6, s=50)
    axes[0, 1].set_title("Scatter Plot: Random Data", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("X values")
    axes[0, 1].set_ylabel("Y values")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Bar chart
    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 78, 32]
    axes[1, 0].bar(categories, values, color="#2ecc71", alpha=0.8, edgecolor="black")
    axes[1, 0].set_title("Bar Chart: Categories", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Category")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Bottom-right: Histogram
    np.random.seed(42)
    histogram_data = np.random.normal(100, 15, 1000)
    axes[1, 1].hist(histogram_data, bins=30, color="#e74c3c", alpha=0.7, edgecolor="black")
    axes[1, 1].set_title("Histogram: Normal Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add overall title
    fig.suptitle("Multi-Panel Figure Demonstration", fontsize=16, fontweight="bold", y=0.995)

    # Adjust spacing to prevent overlap
    plt.tight_layout()

    return fig, axes
