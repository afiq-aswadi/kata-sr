"""Configure Plotly figure layout."""

import plotly.graph_objects as go


def configure_layout(
    fig: go.Figure,
    title: str,
    x_label: str,
    y_label: str,
    showlegend: bool = True,
) -> go.Figure:
    """Configure figure layout with titles, labels, and legend settings.

    Args:
        fig: existing Plotly figure
        title: main plot title
        x_label: x-axis label
        y_label: y-axis label
        showlegend: whether to display the legend

    Returns:
        Figure with updated layout
    """
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=showlegend,
    )
    return fig
