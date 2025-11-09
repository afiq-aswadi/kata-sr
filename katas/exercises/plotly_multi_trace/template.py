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
    # TODO: Create figure, add scatter trace (markers, blue, name="Data Points")
    # and line trace (lines, red, dashed, name="Trend")
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
