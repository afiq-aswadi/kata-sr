"""Add annotation to a specific subplot."""

import plotly.graph_objects as go


def add_annotation_to_subplot(
    fig: go.Figure,
    text: str,
    row: int,
    col: int,
    x: float,
    y: float,
) -> go.Figure:
    """
    Add an annotation to a specific subplot.

    Args:
        fig: Existing figure with subplots
        text: Annotation text
        row: Subplot row (1-indexed)
        col: Subplot column (1-indexed)
        x: X-coordinate in data space
        y: Y-coordinate in data space

    Returns:
        Figure with annotation added

    Behavior:
        - Calculate correct xref/yref based on subplot position
        - For (1,1): xref='x', yref='y'
        - For (1,2): xref='x2', yref='y2'
        - For (2,1): xref='x3', yref='y3', etc. (row-major order)
        - Add annotation with arrow

    Example:
        >>> from plotly.subplots import make_subplots
        >>> fig = make_subplots(rows=2, cols=2)
        >>> fig = add_annotation_to_subplot(fig, "Peak", 1, 1, 2, 5)
        >>> len(fig.layout.annotations) > 0
        True
    """
    # BLANK_START
    raise NotImplementedError(
        "Calculate subplot_index = (row-1)*2 + col, then set xref/yref accordingly"
    )
    # BLANK_END
