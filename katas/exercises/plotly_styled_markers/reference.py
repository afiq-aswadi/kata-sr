"""Create scatter plot with custom per-point marker styling."""

import plotly.graph_objects as go


def create_styled_scatter(
    x: list[float],
    y: list[float],
    colors: list[str] | None = None,
    sizes: list[float] | None = None,
) -> go.Figure:
    """Create scatter plot with variable colors and/or sizes per point.

    Args:
        x: x-axis data points
        y: y-axis data points
        colors: optional list of colors for each point
        sizes: optional list of sizes for each point

    Returns:
        Plotly Figure with styled scatter trace
    """
    marker_dict = {}

    if colors is not None:
        marker_dict["color"] = colors

    if sizes is not None:
        marker_dict["size"] = sizes

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker_dict,
        )
    )
    return fig
