"""Create grouped bar chart with Plotly."""

import plotly.graph_objects as go


def create_grouped_bar_chart(
    categories: list[str],
    group1_values: list[float],
    group2_values: list[float],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> go.Figure:
    """Create grouped bar chart with two groups.

    Args:
        categories: category labels for x-axis
        group1_values: bar heights for first group
        group2_values: bar heights for second group
        group1_name: legend name for first group
        group2_name: legend name for second group

    Returns:
        Figure with two Bar traces displayed side-by-side
    """
    # TODO: Create figure with two Bar traces, each with name and values
    # Both traces should use the same categories for x-axis
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
