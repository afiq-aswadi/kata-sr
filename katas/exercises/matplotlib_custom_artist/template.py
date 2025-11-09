"""Matplotlib custom Artist kata."""

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

        Hints:
            - Check if artist is visible with self.get_visible()
            - Get transform with self.axes.transData
            - Create a Circle patch at self.center with self.radius
            - Draw 8 lines radiating from center to edge (use np.linspace)
            - Use matplotlib.lines.Line2D for lines
        """
        # BLANK_START
        raise NotImplementedError(
            "Implement draw(): check visibility, get transform, draw circle + lines"
        )
        # BLANK_END
