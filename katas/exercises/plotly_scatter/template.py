"""Create basic scatter plot with Plotly graph_objects."""

import plotly.graph_objects as go


def create_scatter_plot(
    x: list[float],
    y: list[float],
    title: str,
    marker_color: str = "blue",
    marker_size: int = 10,
) -> go.Figure:
    """Create a scatter plot with custom markers.

    Args:
        x: x-axis data points
        y: y-axis data points
        title: plot title
        marker_color: color of the markers
        marker_size: size of the markers

    Returns:
        Plotly Figure object with scatter trace
    """
    # TODO: Create figure, add scatter trace with markers, set title
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
