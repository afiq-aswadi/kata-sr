"""Plotly subplots kata - reference solution."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_basic_subplot_grid(
    data: dict[str, list],
) -> go.Figure:
    """Create a 2x2 grid with different trace types."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Scatter", "Bar", "Line", "Heatmap"),
    )

    # Row 1, Col 1: Scatter
    scatter = data["scatter"]
    fig.add_trace(
        go.Scatter(x=scatter["x"], y=scatter["y"], mode="markers", name="Scatter"),
        row=1,
        col=1,
    )

    # Row 1, Col 2: Bar
    bar = data["bar"]
    fig.add_trace(
        go.Bar(x=bar["x"], y=bar["y"], name="Bar"),
        row=1,
        col=2,
    )

    # Row 2, Col 1: Line
    line = data["line"]
    fig.add_trace(
        go.Scatter(x=line["x"], y=line["y"], mode="lines", name="Line"),
        row=2,
        col=1,
    )

    # Row 2, Col 2: Heatmap
    heatmap = data["heatmap"]
    fig.add_trace(
        go.Heatmap(z=heatmap["z"], name="Heatmap"),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=True)
    return fig


def create_shared_axes_figure(
    time_series_data: list[dict],
) -> go.Figure:
    """Create subplots with shared x-axes."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(d["name"] for d in time_series_data),
        vertical_spacing=0.05,
    )

    for i, data in enumerate(time_series_data, start=1):
        fig.add_trace(
            go.Scatter(x=data["x"], y=data["y"], name=data["name"], mode="lines"),
            row=i,
            col=1,
        )

    fig.update_layout(height=600)
    return fig


def create_secondary_yaxis_subplot(
    primary_data: dict,
    secondary_data: dict,
) -> go.Figure:
    """Create a subplot with dual y-axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary y-axis (left)
    fig.add_trace(
        go.Scatter(
            x=primary_data["x"],
            y=primary_data["y"],
            name=primary_data["name"],
            mode="lines",
        ),
        secondary_y=False,
    )

    # Secondary y-axis (right)
    fig.add_trace(
        go.Scatter(
            x=secondary_data["x"],
            y=secondary_data["y"],
            name=secondary_data["name"],
            mode="lines",
        ),
        secondary_y=True,
    )

    # Update y-axis titles
    fig.update_yaxes(title_text=primary_data["name"], secondary_y=False)
    fig.update_yaxes(title_text=secondary_data["name"], secondary_y=True)

    fig.update_layout(height=400)
    return fig


def add_subplot_annotation(
    fig: go.Figure,
    text: str,
    row: int,
    col: int,
    x: float,
    y: float,
) -> go.Figure:
    """Add an annotation to a specific subplot."""
    # Calculate axis reference based on subplot position
    # For a 2x2 grid:
    # (1,1) -> x, y
    # (1,2) -> x2, y2
    # (2,1) -> x3, y3
    # (2,2) -> x4, y4

    # Calculate which subplot this is (1-indexed, row-major order)
    # Assumes 2 columns for now (can be generalized)
    subplot_index = (row - 1) * 2 + col

    if subplot_index == 1:
        xref, yref = "x", "y"
    else:
        xref, yref = f"x{subplot_index}", f"y{subplot_index}"

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


def create_complex_multipanel(
    scatter_data: list[dict],
    bar_data: dict,
    heatmap_z: list[list[float]],
) -> go.Figure:
    """Create a complex multi-panel figure combining multiple features."""
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"secondary_y": True}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=("Dual Y-Axis", "Bar Chart", "Heatmap"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Row 1, Col 1: Dual y-axis subplot
    fig.add_trace(
        go.Scatter(
            x=scatter_data[0]["x"],
            y=scatter_data[0]["y"],
            name=scatter_data[0]["name"],
            mode="lines",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=scatter_data[1]["x"],
            y=scatter_data[1]["y"],
            name=scatter_data[1]["name"],
            mode="lines",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Update y-axes titles for dual axis subplot
    fig.update_yaxes(
        title_text=scatter_data[0]["name"], row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text=scatter_data[1]["name"], row=1, col=1, secondary_y=True
    )

    # Row 1, Col 2: Bar chart
    fig.add_trace(
        go.Bar(x=bar_data["x"], y=bar_data["y"], name="Bar"),
        row=1,
        col=2,
    )

    # Row 2, Col 1-2: Heatmap (spans both columns)
    fig.add_trace(
        go.Heatmap(z=heatmap_z, name="Heatmap", showscale=True),
        row=2,
        col=1,
    )

    fig.update_layout(height=800, title_text="Complex Multi-Panel Figure")
    return fig


def update_individual_subplot_axes(
    fig: go.Figure,
    axis_configs: list[dict],
) -> go.Figure:
    """Update individual subplot axes with custom configurations."""
    for config in axis_configs:
        row = config["row"]
        col = config["col"]
        axis_type = config["axis"]
        title = config["title"]
        axis_range = config.get("range")

        if axis_type == "x":
            fig.update_xaxes(title_text=title, range=axis_range, row=row, col=col)
        elif axis_type == "y":
            fig.update_yaxes(title_text=title, range=axis_range, row=row, col=col)

    return fig
