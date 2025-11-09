"""Create plot with multiple traces using Plotly."""

import plotly.graph_objects as go


def create_multi_trace_plot(
    x: list[float],
    y_scatter: list[float],
    y_line: list[float],
) -> go.Figure:
    """Create a figure with both scatter and line traces.

    Args:
        x: shared x-axis data for both traces
        y_scatter: y-data for scatter trace
        y_line: y-data for line trace

    Returns:
        Figure with two traces: scatter (blue markers) and line (red dashed)
    """
    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_scatter,
            mode="markers",
            name="Data Points",
            marker=dict(color="blue"),
        )
    )

    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color="red", dash="dash"),
        )
    )

    return fig
