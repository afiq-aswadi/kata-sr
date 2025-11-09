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

    Example:
        >>> scatter = {"x": [1, 2, 3], "y": [4, 5, 6]}
        >>> bar = {"x": ["A", "B"], "y": [10, 20]}
        >>> line = {"x": [0, 1], "y": [1, 4]}
        >>> heatmap = {"z": [[1, 2], [3, 4]]}
        >>> fig = create_subplot_grid(scatter, bar, line, heatmap)
        >>> len(fig.data)
        4
    """
    # BLANK_START
    raise NotImplementedError(
        "Use make_subplots(rows=2, cols=2) and add traces with row/col parameters"
    )
    # BLANK_END
