"""Update individual subplot axes with custom configurations."""

import plotly.graph_objects as go


def update_subplot_axes(
    fig: go.Figure,
    row: int,
    col: int,
    x_title: str = None,
    y_title: str = None,
    x_range: list = None,
    y_range: list = None,
) -> go.Figure:
    """
    Update axes for a specific subplot.

    Args:
        fig: Existing figure with subplots
        row: Subplot row (1-indexed)
        col: Subplot column (1-indexed)
        x_title: Optional x-axis title
        y_title: Optional y-axis title
        x_range: Optional x-axis range [min, max]
        y_range: Optional y-axis range [min, max]

    Returns:
        Figure with updated axes
    """
    # Update x-axis if any x properties provided
    if x_title is not None or x_range is not None:
        fig.update_xaxes(
            title_text=x_title,
            range=x_range,
            row=row,
            col=col,
        )

    # Update y-axis if any y properties provided
    if y_title is not None or y_range is not None:
        fig.update_yaxes(
            title_text=y_title,
            range=y_range,
            row=row,
            col=col,
        )

    return fig
