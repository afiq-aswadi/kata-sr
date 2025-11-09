"""Plotly graph_objects basics kata."""

import plotly.graph_objects as go


def create_scatter_plot(x: list[float], y: list[float], title: str) -> go.Figure:
    """Create a scatter plot with custom markers.

    Args:
        x: x-axis data
        y: y-axis data
        title: plot title

    Returns:
        Plotly Figure with scatter trace
    """
    # TODO: Create a Figure with a Scatter trace
    # - Use mode='markers'
    # - Set marker size to 10 and color to 'blue'
    # - Add title to layout
    # BLANK_START
    pass
    # BLANK_END


def create_line_plot(
    x: list[float], y: list[float], line_name: str, line_color: str = "red"
) -> go.Figure:
    """Create a line plot.

    Args:
        x: x-axis data
        y: y-axis data
        line_name: name for legend
        line_color: line color

    Returns:
        Plotly Figure with line trace
    """
    # TODO: Create a Figure with a Scatter trace in line mode
    # - Use mode='lines'
    # - Set line color and name
    # BLANK_START
    pass
    # BLANK_END


def create_bar_chart(
    categories: list[str], values: list[float], title: str
) -> go.Figure:
    """Create a bar chart.

    Args:
        categories: category names for x-axis
        values: bar heights
        title: chart title

    Returns:
        Plotly Figure with bar trace
    """
    # TODO: Create a Figure with a Bar trace
    # - Use go.Bar with x=categories, y=values
    # - Add title to layout
    # BLANK_START
    pass
    # BLANK_END


def add_custom_hover_template(fig: go.Figure, template: str) -> go.Figure:
    """Add custom hover template to all traces in figure.

    Args:
        fig: existing figure
        template: hover template string

    Returns:
        Figure with updated hover template
    """
    # TODO: Update all traces with the custom hover template
    # Hint: use fig.update_traces(hovertemplate=template)
    # BLANK_START
    pass
    # BLANK_END


def create_multi_trace_plot(
    x: list[float], y1: list[float], y2: list[float]
) -> go.Figure:
    """Create a plot with multiple traces.

    Args:
        x: shared x-axis data
        y1: first dataset (scatter)
        y2: second dataset (line)

    Returns:
        Figure with two traces: scatter and line
    """
    # TODO: Create a figure with two traces
    # Trace 1: Scatter with mode='markers', name='Data Points', color='blue'
    # Trace 2: Scatter with mode='lines', name='Trend', color='red', dash='dash'
    # BLANK_START
    pass
    # BLANK_END


def configure_layout(
    fig: go.Figure,
    title: str,
    x_label: str,
    y_label: str,
    showlegend: bool = True,
) -> go.Figure:
    """Configure figure layout with titles and labels.

    Args:
        fig: existing figure
        title: main title
        x_label: x-axis label
        y_label: y-axis label
        showlegend: whether to show legend

    Returns:
        Figure with updated layout
    """
    # TODO: Update layout with title, axis labels, and legend settings
    # Hint: use fig.update_layout()
    # BLANK_START
    pass
    # BLANK_END


def create_styled_scatter(
    x: list[float],
    y: list[float],
    colors: list[str] | None = None,
    sizes: list[float] | None = None,
) -> go.Figure:
    """Create scatter plot with custom styling per point.

    Args:
        x: x-axis data
        y: y-axis data
        colors: color for each point (optional)
        sizes: size for each point (optional)

    Returns:
        Figure with styled scatter trace
    """
    # TODO: Create scatter plot with variable colors and sizes
    # - If colors provided, set marker.color = colors
    # - If sizes provided, set marker.size = sizes
    # - Use mode='markers'
    # BLANK_START
    pass
    # BLANK_END


def add_hover_formatting(x: list[float], y: list[float], names: list[str]) -> go.Figure:
    """Create scatter plot with formatted hover text.

    Args:
        x: x-axis data
        y: y-axis data
        names: labels for each point

    Returns:
        Figure with formatted hover information
    """
    # TODO: Create scatter plot with custom hover template
    # - Show point name, x value, and y value (formatted to 2 decimals)
    # - Template format: '<b>%{text}</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>'
    # - Use text parameter to pass names
    # BLANK_START
    pass
    # BLANK_END


def create_grouped_bar_chart(
    categories: list[str],
    group1_values: list[float],
    group2_values: list[float],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> go.Figure:
    """Create a grouped bar chart with two groups.

    Args:
        categories: x-axis categories
        group1_values: values for first group
        group2_values: values for second group
        group1_name: legend name for first group
        group2_name: legend name for second group

    Returns:
        Figure with two bar traces
    """
    # TODO: Create figure with two Bar traces
    # - Each trace should have name and values
    # - Both traces use same categories for x-axis
    # BLANK_START
    pass
    # BLANK_END
