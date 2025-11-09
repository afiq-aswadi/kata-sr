"""Create vertically stacked subplots with shared x-axes."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_shared_axes_plot(time_series: list[dict]) -> go.Figure:
    """
    Create vertically stacked line plots with shared x-axes.

    Args:
        time_series: List of dicts, each with 'x', 'y', and 'name' keys
                     Each dict represents one subplot

    Returns:
        Figure with vertically stacked subplots sharing x-axis
    """
    n_plots = len(time_series)

    # Create vertical stack with shared x-axes
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(d["name"] for d in time_series),
        vertical_spacing=0.05,
    )

    # Add one trace to each row
    for i, data in enumerate(time_series, start=1):
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                name=data["name"],
                mode="lines",
            ),
            row=i,
            col=1,
        )

    fig.update_layout(height=200 * n_plots + 100)
    return fig
