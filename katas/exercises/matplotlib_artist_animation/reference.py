"""Matplotlib Artist API and FuncAnimation kata - reference solution."""

import tempfile

import matplotlib.animation as animation
import matplotlib.artist as artist
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class CustomCircleArtist(artist.Artist):
    """Custom Artist that draws a circle with decorative patterns.

    This demonstrates the Artist API by creating a custom drawable element
    that can be added to matplotlib axes.
    """

    def __init__(self, center, radius, color="blue"):
        """Initialize the custom artist.

        Args:
            center: (x, y) tuple for circle center
            radius: circle radius
            color: circle color
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color

    def draw(self, renderer):
        """Draw the custom artist using the renderer.

        This is the core method that must be implemented by all Artists.
        It uses the transform stack to convert from data to display coordinates.

        Args:
            renderer: matplotlib renderer object
        """
        if not self.get_visible():
            return

        # Get transform from data to display coordinates
        transform = self.axes.transData

        # Create a circle patch
        circle = patches.Circle(
            self.center, self.radius, color=self.color, alpha=0.6, transform=transform
        )

        # Draw the circle
        circle.draw(renderer)

        # Draw decorative lines
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x = self.center[0] + self.radius * np.cos(angle)
            y = self.center[1] + self.radius * np.sin(angle)
            line_x = [self.center[0], x]
            line_y = [self.center[1], y]

            # Transform coordinates
            transformed = transform.transform(np.column_stack([line_x, line_y]))

            # Draw line using the renderer
            renderer.draw_path(
                renderer.new_gc(),
                renderer.new_gc()._renderer.get_path_collection_extents(
                    transform, [line_x], [line_y]
                )[0],
                transform,
                renderer.new_gc()._rgb,
            )


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
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + frame / 10)
        line.set_data(x, y)
        return [line]

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval)

    return anim


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


def create_blitting_animation(num_frames=100, interval=33):
    """Create an optimized animation using blitting.

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


def save_animation_to_file(anim, filename=None, fps=20, dpi=100):
    """Save animation to a file.

    Args:
        anim: FuncAnimation object
        filename: output filename (None for temp file)
        fps: frames per second
        dpi: dots per inch for output

    Returns:
        path to saved file
    """
    if filename is None:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, mode="w"
        )
        filename = temp_file.name
        temp_file.close()

    # Save animation - pillow writer works without ffmpeg
    anim.save(filename, writer="pillow", fps=fps, dpi=dpi)

    return filename


def create_interactive_animation(data_source):
    """Create an animation that responds to data updates.

    This demonstrates how to create animations that can be updated
    based on external data or user interactions.

    Args:
        data_source: callable that returns (x, y) data for given frame

    Returns:
        FuncAnimation object
    """
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], "m-", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Interactive Data Animation")
    ax.grid(True)

    def init():
        line.set_data([], [])
        return [line]

    def update(frame):
        """Get data from external source and update plot."""
        x, y = data_source(frame)
        line.set_data(x, y)

        # Dynamically adjust limits if needed
        if len(x) > 0:
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y) - 0.5, max(y) + 0.5)

        return [line]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=100, interval=50, blit=True
    )

    return anim


def add_custom_artist_to_plot():
    """Demonstrate adding a custom Artist to a plot.

    Returns:
        tuple of (figure, axes, custom_artist)
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Custom Artist Example")

    # Create and add custom artist
    custom_artist = CustomCircleArtist(center=(5, 5), radius=2, color="purple")
    custom_artist.set_axes(ax)
    ax.add_artist(custom_artist)

    return fig, ax, custom_artist
