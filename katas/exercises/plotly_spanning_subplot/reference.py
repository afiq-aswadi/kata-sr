"""Create a subplot that spans multiple columns."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_spanning_subplot(
    top_data: list[dict],
    bottom_data: dict,
) -> go.Figure:
    """
    Create a 2-row layout where bottom row spans both columns.

    Args:
        top_data: List of 2 dicts with 'x', 'y' for top-left and top-right scatter plots
        bottom_data: Dict with 'z' (2D array) for bottom heatmap

    Returns:
        Figure with spanning bottom subplot
    """
    # Create layout with spanning bottom row
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{}, {}],  # Top row: 2 regular cells
            [{"colspan": 2}, None],  # Bottom row: spanning cell + None
        ],
        subplot_titles=("Top Left", "Top Right", "Bottom Spanning"),
        vertical_spacing=0.15,
    )

    # Add top-left scatter
    fig.add_trace(
        go.Scatter(
            x=top_data[0]["x"],
            y=top_data[0]["y"],
            mode="markers",
            name="Top Left",
        ),
        row=1,
        col=1,
    )

    # Add top-right scatter
    fig.add_trace(
        go.Scatter(
            x=top_data[1]["x"],
            y=top_data[1]["y"],
            mode="markers",
            name="Top Right",
        ),
        row=1,
        col=2,
    )

    # Add bottom spanning heatmap
    fig.add_trace(
        go.Heatmap(z=bottom_data["z"], name="Heatmap", showscale=True),
        row=2,
        col=1,  # Spanning cell starts at col 1
    )

    fig.update_layout(height=600)
    return fig
