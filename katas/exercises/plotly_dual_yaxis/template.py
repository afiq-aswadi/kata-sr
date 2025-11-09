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

    Behavior:
        - Primary trace uses left y-axis
        - Secondary trace uses right y-axis
        - Both traces share the same x-axis
        - Y-axis titles should match trace names

    Example:
        >>> primary = {"x": [1,2,3], "y": [10,20,30], "name": "Temperature"}
        >>> secondary = {"x": [1,2,3], "y": [100,200,150], "name": "Pressure"}
        >>> fig = create_dual_yaxis_plot(primary, secondary)
        >>> len(fig.data)
        2
    """
    # BLANK_START
    raise NotImplementedError(
        "Use make_subplots with specs=[[{'secondary_y': True}]] and add traces with secondary_y parameter"
    )
    # BLANK_END
