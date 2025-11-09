"""Matplotlib simple animation kata - reference solution."""

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
    """
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], "r-", linewidth=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x + phase)")
    ax.set_title("Moving Sine Wave")

    def update(frame):
        """Update function called for each frame."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + frame / 10)
        line.set_data(x, y)
        return [line]

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval)

    return anim
