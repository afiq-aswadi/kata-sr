"""Create a subplot with dual y-axes."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dual_yaxis_plot(
    primary_data: dict,
    secondary_data: dict,
) -> go.Figure:
    """
    Create a subplot with two y-axes (primary and secondary).

    Args:
        primary_data: Dict with 'x', 'y', 'name' for left y-axis
        secondary_data: Dict with 'x', 'y', 'name' for right y-axis

    Returns:
        Figure with dual y-axes
    """
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add primary trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=primary_data["x"],
            y=primary_data["y"],
            name=primary_data["name"],
            mode="lines",
        ),
        secondary_y=False,
    )

    # Add secondary trace (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=secondary_data["x"],
            y=secondary_data["y"],
            name=secondary_data["name"],
            mode="lines",
        ),
        secondary_y=True,
    )

    # Update y-axis titles
    fig.update_yaxes(title_text=primary_data["name"], secondary_y=False)
    fig.update_yaxes(title_text=secondary_data["name"], secondary_y=True)

    fig.update_layout(height=400)
    return fig
