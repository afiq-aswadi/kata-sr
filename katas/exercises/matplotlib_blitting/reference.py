"""Matplotlib blitting optimization kata - reference solution."""

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
    """
    fig, ax = plt.subplots()

    # Create multiple animated elements
    (line1,) = ax.plot([], [], "r-", linewidth=2, label="sin(x)")
    (line2,) = ax.plot([], [], "b-", linewidth=2, label="cos(x)")
    (point,) = ax.plot([], [], "go", markersize=10, label="moving point")

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-2, 2)
    ax.legend(loc="upper right")
    ax.grid(True)

    def init():
        """Initialize all animated elements."""
        line1.set_data([], [])
        line2.set_data([], [])
        point.set_data([], [])
        return [line1, line2, point]

    def update(frame):
        """Update all animated elements - only these will be redrawn."""
        x = np.linspace(0, 2 * np.pi, 100)

        # Update sine wave
        y1 = np.sin(x + frame / 10)
        line1.set_data(x, y1)

        # Update cosine wave
        y2 = np.cos(x + frame / 10)
        line2.set_data(x, y2)

        # Update moving point
        point_x = frame / num_frames * 2 * np.pi
        point_y = np.sin(point_x + frame / 10)
        point.set_data([point_x], [point_y])

        return [line1, line2, point]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=num_frames, interval=interval, blit=True
    )

    return anim
