"""Plotly subplots kata - multi-panel layouts."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_basic_subplot_grid(
    data: dict[str, list],
) -> go.Figure:
    """Create a 2x2 grid with different trace types.

    Args:
        data: dictionary with keys 'scatter', 'bar', 'line', 'heatmap'
              each containing appropriate data for that trace type

    Returns:
        Figure with 2x2 subplot grid

    Expected layout:
        Row 1, Col 1: Scatter plot
        Row 1, Col 2: Bar chart
        Row 2, Col 1: Line plot
        Row 2, Col 2: Heatmap
    """
    # TODO: Create 2x2 subplot grid with subplot_titles
    # Hint: make_subplots(rows=2, cols=2, subplot_titles=(...))
    # BLANK_START
    pass
    # BLANK_END


def create_shared_axes_figure(
    time_series_data: list[dict],
) -> go.Figure:
    """Create subplots with shared x-axes.

    Args:
        time_series_data: list of dicts, each with 'x', 'y', and 'name' keys

    Returns:
        Figure with vertically stacked subplots sharing x-axis

    Should create 3 rows x 1 column with shared x-axes so zooming
    on one subplot affects all subplots.
    """
    # TODO: Create 3x1 subplot grid with shared_xaxes=True
    # TODO: Add one scatter trace to each row
    # Hint: fig.add_trace(..., row=i, col=1)
    # BLANK_START
    pass
    # BLANK_END


def create_secondary_yaxis_subplot(
    primary_data: dict,
    secondary_data: dict,
) -> go.Figure:
    """Create a subplot with dual y-axes.

    Args:
        primary_data: dict with 'x', 'y', 'name' for primary y-axis
        secondary_data: dict with 'x', 'y', 'name' for secondary y-axis

    Returns:
        Figure with one subplot containing two y-axes

    The primary trace should use the left y-axis,
    the secondary trace should use the right y-axis.
    """
    # TODO: Create subplot with specs=[[{"secondary_y": True}]]
    # TODO: Add primary trace with secondary_y=False
    # TODO: Add secondary trace with secondary_y=True
    # TODO: Update y-axis titles for both axes
    # Hint: fig.update_yaxes(title_text="...", secondary_y=True/False)
    # BLANK_START
    pass
    # BLANK_END


def add_subplot_annotation(
    fig: go.Figure,
    text: str,
    row: int,
    col: int,
    x: float,
    y: float,
) -> go.Figure:
    """Add an annotation to a specific subplot.

    Args:
        fig: existing figure with subplots
        text: annotation text
        row: subplot row (1-indexed)
        col: subplot column (1-indexed)
        x: x-coordinate in data space
        y: y-coordinate in data space

    Returns:
        Figure with annotation added

    The annotation should be positioned relative to the specified subplot's axes.
    """
    # TODO: Add annotation with correct xref and yref
    # Hint: for subplot (1,1) use xref='x', yref='y'
    # Hint: for subplot (1,2) use xref='x2', yref='y2', etc.
    # Hint: calculate axis reference based on subplot position
    # BLANK_START
    pass
    # BLANK_END


def create_complex_multipanel(
    scatter_data: list[dict],
    bar_data: dict,
    heatmap_z: list[list[float]],
) -> go.Figure:
    """Create a complex multi-panel figure combining multiple features.

    Args:
        scatter_data: list of 2 dicts with 'x', 'y', 'name' for dual y-axis subplot
        bar_data: dict with 'x', 'y' for bar chart
        heatmap_z: 2D array for heatmap

    Returns:
        Figure with complex layout

    Layout should be:
        - Row 1, Col 1: Dual y-axis subplot (2 scatter traces)
        - Row 1, Col 2: Bar chart
        - Row 2, Col 1-2: Heatmap spanning 2 columns

    The dual y-axis subplot should have:
        - First scatter on primary (left) y-axis
        - Second scatter on secondary (right) y-axis
        - Different y-axis titles for each
    """
    # TODO: Create 2x2 grid with custom specs
    # Hint: specs=[[{"secondary_y": True}, {}],
    #              [{"colspan": 2}, None]]
    # TODO: Add scatter traces to (1,1) with secondary_y parameter
    # TODO: Add bar trace to (1,2)
    # TODO: Add heatmap to (2,1) - it will span both columns
    # TODO: Update layout with title and appropriate height
    # BLANK_START
    pass
    # BLANK_END


def update_individual_subplot_axes(
    fig: go.Figure,
    axis_configs: list[dict],
) -> go.Figure:
    """Update individual subplot axes with custom configurations.

    Args:
        fig: existing figure with subplots
        axis_configs: list of dicts with keys:
            - 'row': subplot row
            - 'col': subplot column
            - 'axis': 'x' or 'y'
            - 'title': axis title
            - 'range': optional [min, max] for axis range

    Returns:
        Figure with updated axes

    Should use update_xaxes() or update_yaxes() with row/col parameters.
    """
    # TODO: Iterate through axis_configs
    # TODO: Use update_xaxes() or update_yaxes() with row and col
    # Hint: fig.update_xaxes(title=..., range=..., row=row, col=col)
    # BLANK_START
    pass
    # BLANK_END
