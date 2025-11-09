"""Matplotlib figure/axes kata - master the OO interface for multi-panel plots.

Your task: Create a 2x2 multi-panel plot with proper styling and customization.

Requirements:
1. Create a figure with 2x2 subplots (4 total subplots)
2. Use figsize=(12, 10) for proper sizing
3. Create four different plot types:
   - Top-left: Line plot (sin and cos)
   - Top-right: Scatter plot
   - Bottom-left: Bar chart
   - Bottom-right: Histogram
4. Each subplot must have:
   - Descriptive title with fontsize=12 and fontweight="bold"
   - X and Y axis labels
   - Grid with alpha=0.3
5. Line plot must have legend in upper right
6. Add figure suptitle "Multi-Panel Figure Demonstration" (fontsize=16, fontweight="bold", y=0.995)
7. Use tight_layout() to prevent overlap

Hints:
- Use fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
- Access individual subplots with axes[row, col]
- Use ax.set_title(), ax.set_xlabel(), ax.set_ylabel(), not plt.title()
- Generate data with numpy (already imported)
"""

import matplotlib.pyplot as plt
import numpy as np


def create_multi_panel_plot():
    """Create a 2x2 multi-panel plot demonstrating the figure/axes pattern.

    Returns:
        tuple: (fig, axes) where fig is the Figure object and axes is a 2x2 ndarray of Axes
    """
    # TODO: Create figure with 2x2 subplots, figsize=(12, 10)
    # fig, axes = plt.subplots(...)

    # Generate data for plots
    x = np.linspace(0, 2 * np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # TODO: Top-left subplot (axes[0, 0]): Line plot
    # Plot sin(x) as red solid line ("r-") with label="sin(x)", linewidth=2
    # Plot cos(x) as blue dashed line ("b--") with label="cos(x)", linewidth=2
    # Set title: "Line Plot: Trigonometric Functions" (fontsize=12, fontweight="bold")
    # Set xlabel: "x (radians)"
    # Set ylabel: "y"
    # Add legend at upper right
    # Add grid with alpha=0.3

    # Generate random data for scatter plot
    np.random.seed(42)
    scatter_x = np.random.randn(100)
    scatter_y = np.random.randn(100)

    # TODO: Top-right subplot (axes[0, 1]): Scatter plot
    # Create scatter plot with blue circles, alpha=0.6, marker="o", s=50
    # Set title: "Scatter Plot: Random Data" (fontsize=12, fontweight="bold")
    # Set xlabel: "X values"
    # Set ylabel: "Y values"
    # Add grid with alpha=0.3

    # Data for bar chart
    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 78, 32]

    # TODO: Bottom-left subplot (axes[1, 0]): Bar chart
    # Create bar chart with color="#2ecc71", alpha=0.8, edgecolor="black"
    # Set title: "Bar Chart: Categories" (fontsize=12, fontweight="bold")
    # Set xlabel: "Category"
    # Set ylabel: "Value"
    # Add grid with alpha=0.3, axis="y"

    # Generate data for histogram
    np.random.seed(42)
    histogram_data = np.random.normal(100, 15, 1000)

    # TODO: Bottom-right subplot (axes[1, 1]): Histogram
    # Create histogram with 30 bins, color="#e74c3c", alpha=0.7, edgecolor="black"
    # Set title: "Histogram: Normal Distribution" (fontsize=12, fontweight="bold")
    # Set xlabel: "Value"
    # Set ylabel: "Frequency"
    # Add grid with alpha=0.3, axis="y"

    # TODO: Add figure suptitle "Multi-Panel Figure Demonstration"
    # Use fontsize=16, fontweight="bold", y=0.995

    # TODO: Call plt.tight_layout() to prevent overlap

    # TODO: Return the figure and axes
    # return fig, axes
