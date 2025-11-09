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

    Behavior:
        - Use update_xaxes() and update_yaxes() with row/col parameters
        - Only update properties that are provided (not None)
        - Leave other subplots unchanged

    Example:
        >>> from plotly.subplots import make_subplots
        >>> fig = make_subplots(rows=2, cols=2)
        >>> fig = update_subplot_axes(fig, 1, 1, x_title="Time", x_range=[0, 10])
        >>> fig.layout.xaxis.title.text
        'Time'
    """
    # BLANK_START
    raise NotImplementedError(
        "Use fig.update_xaxes() and fig.update_yaxes() with row, col parameters"
    )
    # BLANK_END
