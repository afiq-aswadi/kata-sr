"""Matplotlib blitting optimization kata."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_blitting_animation(num_frames=100, interval=33):
    """Create an optimized animation using blitting with multiple elements.

    Blitting only redraws the parts of the figure that have changed,
    significantly improving performance for complex animations.

    Args:
        num_frames: number of animation frames
        interval: milliseconds between frames (33ms ≈ 30 fps)

    Returns:
        FuncAnimation object

    Hints:
        - Create figure and axes
        - Create 3 line artists: line1 (sin, red), line2 (cos, blue), point (green)
        - Set axis limits, legend, and grid
        - Define init() that:
          * Sets all elements to empty data
          * Returns [line1, line2, point]
        - Define update(frame) that:
          * Updates line1 with sin(x + phase)
          * Updates line2 with cos(x + phase)
          * Updates point position along sine curve
          * Returns [line1, line2, point]
        - Create FuncAnimation with blit=True
        - The key: both init and update must return the list of artists
    """
    # BLANK_START
    raise NotImplementedError(
        "Create figure with 3 animated elements, init/update functions, blit=True"
    )
    # BLANK_END
