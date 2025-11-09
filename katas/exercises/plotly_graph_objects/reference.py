"""Plotly graph_objects basics kata - reference solution."""

import plotly.graph_objects as go


def create_scatter_plot(x: list[float], y: list[float], title: str) -> go.Figure:
    """Create a scatter plot with custom markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10, color="blue"),
        )
    )
    fig.update_layout(title=title)
    return fig


def create_line_plot(
    x: list[float], y: list[float], line_name: str, line_color: str = "red"
) -> go.Figure:
    """Create a line plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=line_name,
            line=dict(color=line_color),
        )
    )
    return fig


def create_bar_chart(
    categories: list[str], values: list[float], title: str
) -> go.Figure:
    """Create a bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values))
    fig.update_layout(title=title)
    return fig


def add_custom_hover_template(fig: go.Figure, template: str) -> go.Figure:
    """Add custom hover template to all traces in figure."""
    fig.update_traces(hovertemplate=template)
    return fig


def create_multi_trace_plot(
    x: list[float], y1: list[float], y2: list[float]
) -> go.Figure:
    """Create a plot with multiple traces."""
    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            mode="markers",
            name="Data Points",
            marker=dict(color="blue"),
        )
    )

    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            mode="lines",
            name="Trend",
            line=dict(color="red", dash="dash"),
        )
    )

    return fig


def configure_layout(
    fig: go.Figure,
    title: str,
    x_label: str,
    y_label: str,
    showlegend: bool = True,
) -> go.Figure:
    """Configure figure layout with titles and labels."""
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=showlegend,
    )
    return fig


def create_styled_scatter(
    x: list[float],
    y: list[float],
    colors: list[str] | None = None,
    sizes: list[float] | None = None,
) -> go.Figure:
    """Create scatter plot with custom styling per point."""
    marker_dict = {}

    if colors is not None:
        marker_dict["color"] = colors

    if sizes is not None:
        marker_dict["size"] = sizes

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker_dict,
        )
    )
    return fig


def add_hover_formatting(x: list[float], y: list[float], names: list[str]) -> go.Figure:
    """Create scatter plot with formatted hover text."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=names,
            hovertemplate="<b>%{text}</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>",
        )
    )
    return fig


def create_grouped_bar_chart(
    categories: list[str],
    group1_values: list[float],
    group2_values: list[float],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> go.Figure:
    """Create a grouped bar chart with two groups."""
    fig = go.Figure()

    fig.add_trace(go.Bar(x=categories, y=group1_values, name=group1_name))

    fig.add_trace(go.Bar(x=categories, y=group2_values, name=group2_name))

    return fig
