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

    Layout:
        Row 1, Col 1: Scatter plot (top_data[0])
        Row 1, Col 2: Scatter plot (top_data[1])
        Row 2, Col 1-2: Heatmap spanning both columns (bottom_data)

    Example:
        >>> top = [{"x": [1,2], "y": [3,4]}, {"x": [5,6], "y": [7,8]}]
        >>> bottom = {"z": [[1,2,3], [4,5,6]]}
        >>> fig = create_spanning_subplot(top, bottom)
        >>> len(fig.data)
        3
    """
    # BLANK_START
    raise NotImplementedError(
        "Use specs=[[{}, {}], [{'colspan': 2}, None]] and add heatmap to row=2, col=1"
    )
    # BLANK_END
