"""Create scatter plot with formatted hover text."""

import plotly.graph_objects as go


def create_formatted_hover_scatter(
    x: list[float],
    y: list[float],
    labels: list[str],
) -> go.Figure:
    """Create scatter plot with custom formatted hover text.

    Args:
        x: x-axis data points
        y: y-axis data points
        labels: text labels for each point (shown in hover)

    Returns:
        Figure with scatter trace and formatted hover template
    """
    # TODO: Create scatter plot with:
    # - text parameter set to labels
    # - hovertemplate showing label (bold), x value, and y value (2 decimals)
    # Format: '<b>%{text}</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>'
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
