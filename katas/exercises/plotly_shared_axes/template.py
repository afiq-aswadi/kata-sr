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

    Behavior:
        - Create N rows x 1 column layout (N = len(time_series))
        - All subplots share the same x-axis (zooming one affects all)
        - Each subplot shows a separate time series

    Example:
        >>> data = [
        ...     {"x": [1,2,3], "y": [10,20,15], "name": "Series 1"},
        ...     {"x": [1,2,3], "y": [5,15,10], "name": "Series 2"}
        ... ]
        >>> fig = create_shared_axes_plot(data)
        >>> len(fig.data)
        2
    """
    # BLANK_START
    raise NotImplementedError(
        "Use make_subplots with shared_xaxes=True and add traces to each row"
    )
    # BLANK_END
