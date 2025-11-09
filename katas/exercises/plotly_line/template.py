"""Create basic line plot with Plotly graph_objects."""

import plotly.graph_objects as go


def create_line_plot(
    x: list[float],
    y: list[float],
    line_name: str,
    line_color: str = "blue",
) -> go.Figure:
    """Create a line plot with custom styling.

    Args:
        x: x-axis data points
        y: y-axis data points
        line_name: name for the line (shown in legend)
        line_color: color of the line

    Returns:
        Plotly Figure object with line trace
    """
    # TODO: Create figure with line trace (mode='lines')
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
