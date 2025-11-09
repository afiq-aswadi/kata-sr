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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=marker_size, color=marker_color),
        )
    )
    fig.update_layout(title=title)
    return fig
