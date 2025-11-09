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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=line_name,
            line=dict(color=line_color),
        )
    )
    return fig
