"""Create a 2x2 subplot grid with different trace types."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_subplot_grid(
    scatter_data: dict,
    bar_data: dict,
    line_data: dict,
    heatmap_data: dict,
) -> go.Figure:
    """
    Create a 2x2 grid with scatter, bar, line, and heatmap traces.

    Args:
        scatter_data: Dict with 'x' and 'y' keys for scatter plot
        bar_data: Dict with 'x' and 'y' keys for bar chart
        line_data: Dict with 'x' and 'y' keys for line plot
        heatmap_data: Dict with 'z' key (2D array) for heatmap

    Returns:
        Figure with 2x2 subplot grid containing all traces

    Layout:
        (1,1): Scatter plot
        (1,2): Bar chart
        (2,1): Line plot
        (2,2): Heatmap
    """
    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Scatter", "Bar", "Line", "Heatmap"),
    )

    # Row 1, Col 1: Scatter
    fig.add_trace(
        go.Scatter(
            x=scatter_data["x"],
            y=scatter_data["y"],
            mode="markers",
            name="Scatter",
        ),
        row=1,
        col=1,
    )

    # Row 1, Col 2: Bar
    fig.add_trace(
        go.Bar(x=bar_data["x"], y=bar_data["y"], name="Bar"),
        row=1,
        col=2,
    )

    # Row 2, Col 1: Line
    fig.add_trace(
        go.Scatter(
            x=line_data["x"],
            y=line_data["y"],
            mode="lines",
            name="Line",
        ),
        row=2,
        col=1,
    )

    # Row 2, Col 2: Heatmap
    fig.add_trace(
        go.Heatmap(z=heatmap_data["z"], name="Heatmap"),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=True)
    return fig
