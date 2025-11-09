"""Matplotlib animation with init function kata - reference solution."""

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
    """
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], "b-", linewidth=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sine Wave Animation with Init")

    def init():
        """Initialize animation - set up blank canvas."""
        line.set_data([], [])
        return [line]

    def update(frame):
        """Update animation for each frame."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + frame / 10)
        line.set_data(x, y)
        return [line]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=num_frames, interval=interval, blit=blit
    )

    return anim
