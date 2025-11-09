"""Matplotlib simple animation kata."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_simple_animation(num_frames=100, interval=50):
    """Create a simple sine wave animation.

    Args:
        num_frames: number of animation frames
        interval: milliseconds between frames

    Returns:
        FuncAnimation object

    Hints:
        - Create figure and axes with plt.subplots()
        - Initialize empty line with ax.plot([], [], 'r-')
        - Set axis limits and labels
        - Define update(frame) function that:
          * Creates x from 0 to 2π
          * Computes y = sin(x + phase) where phase depends on frame
          * Updates line with line.set_data(x, y)
          * Returns [line]
        - Create FuncAnimation(fig, update, frames=num_frames, interval=interval)
    """
    # BLANK_START
    raise NotImplementedError(
        "Create figure, line, update function, and FuncAnimation"
    )
    # BLANK_END
