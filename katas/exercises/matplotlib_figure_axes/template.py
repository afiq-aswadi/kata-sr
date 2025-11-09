"""Matplotlib figure/axes kata - master the OO interface for plotting.

Learn the fundamental matplotlib pattern: fig, ax = plt.subplots()

This is the foundation for all matplotlib plotting. Instead of using the pyplot
state machine (plt.plot, plt.xlabel, etc.), you'll use the object-oriented
interface which gives you explicit control and is essential for creating
publication-quality figures.

Key concepts:
- Create figure and axes explicitly with plt.subplots()
- Use ax.plot(), ax.set_xlabel(), ax.set_title() (not plt.*)
- Customize plots with labels, legends, grid, and styling
- Return fig and ax for further manipulation

Requirements:
- Create figure with size 10x6 inches
- Plot sin(x) and cos(x) from 0 to 2π
- sin(x): red solid line ("r-"), linewidth=2, label="sin(x)"
- cos(x): blue dashed line ("b--"), linewidth=2, label="cos(x)"
- Title: "Trigonometric Functions" (fontsize=14, fontweight="bold")
- X-axis label: "x (radians)" (fontsize=12)
- Y-axis label: "y" (fontsize=12)
- Legend at upper right
- Grid with alpha=0.3
- Call tight_layout()

Hints:
- Use np.linspace(0, 2 * np.pi, 100) to generate x values
- Use np.sin(x) and np.cos(x) for y values
- The pattern is: fig, ax = plt.subplots(figsize=(w, h))
- Style format: "color+linestyle" like "r-" or "b--"
"""

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
    raise NotImplementedError("Implement line plot using fig, ax = plt.subplots() pattern")
    # BLANK_END
