"""Create basic bar chart with Plotly graph_objects."""

import plotly.graph_objects as go


def create_bar_chart(
    categories: list[str],
    values: list[float],
    title: str,
) -> go.Figure:
    """Create a bar chart for categorical data.

    Args:
        categories: category names for x-axis
        values: bar heights for each category
        title: chart title

    Returns:
        Plotly Figure object with bar trace
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values))
    fig.update_layout(title=title)
    return fig
