"""Matplotlib custom Artist kata - reference solution."""

import matplotlib.artist as artist
import matplotlib.patches as patches
import numpy as np


class CustomCircleArtist(artist.Artist):
    """Custom Artist that draws a circle with decorative radiating lines.

    This demonstrates the matplotlib Artist API by creating a custom
    drawable element that can be added to axes.
    """

    def __init__(self, center, radius, color="blue"):
        """Initialize the custom artist.

        Args:
            center: (x, y) tuple for circle center in data coordinates
            radius: circle radius in data coordinates
            color: circle color
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color

    def draw(self, renderer):
        """Draw the custom artist using the renderer.

        This is the core method that matplotlib calls to render the artist.
        It uses the transform stack to convert from data to display coordinates.

        Args:
            renderer: matplotlib renderer object
        """
        if not self.get_visible():
            return

        # Get transform from data to display coordinates
        transform = self.axes.transData

        # Create and draw a circle patch
        circle = patches.Circle(
            self.center, self.radius, color=self.color, alpha=0.6, transform=transform
        )
        circle.draw(renderer)

        # Draw decorative lines from center to edge
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x_end = self.center[0] + self.radius * np.cos(angle)
            y_end = self.center[1] + self.radius * np.sin(angle)

            # Create line patch
            from matplotlib.lines import Line2D

            line = Line2D(
                [self.center[0], x_end],
                [self.center[1], y_end],
                color=self.color,
                linewidth=1,
                transform=transform,
            )
            line.draw(renderer)
