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
    """
    # Calculate subplot index (assumes 2 columns)
    # (1,1)→1, (1,2)→2, (2,1)→3, (2,2)→4
    subplot_index = (row - 1) * 2 + col

    # Determine axis references
    if subplot_index == 1:
        xref, yref = "x", "y"
    else:
        xref, yref = f"x{subplot_index}", f"y{subplot_index}"

    # Add annotation
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        xref=xref,
        yref=yref,
        showarrow=True,
        arrowhead=2,
    )

    return fig
