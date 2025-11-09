"""Matplotlib animation with init function kata."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_animation_with_init(num_frames=100, interval=50, blit=True):
    """Create animation with both init_func and update_func.

    The init_func sets up the plot background, and update draws the changing elements.
    This pattern is essential for blitting.

    Args:
        num_frames: number of animation frames
        interval: milliseconds between frames
        blit: whether to use blitting for performance

    Returns:
        FuncAnimation object

    Hints:
        - Create figure, axes, and empty line
        - Define init() function that:
          * Sets line to empty data []
          * Returns [line]
        - Define update(frame) function that:
          * Creates x and y data
          * Updates line with set_data()
          * Returns [line]
        - Create FuncAnimation with init_func parameter
        - Pass blit parameter to FuncAnimation
    """
    # BLANK_START
    raise NotImplementedError(
        "Create figure, init() and update() functions, FuncAnimation with init_func"
    )
    # BLANK_END
