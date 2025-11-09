"""Apply custom hover template to Plotly figure."""

import plotly.graph_objects as go


def add_hover_template(
    fig: go.Figure,
    template: str,
) -> go.Figure:
    """Apply custom hover template to all traces in figure.

    Args:
        fig: existing Plotly figure
        template: hover template string (HTML-like format)

    Returns:
        Figure with updated hover template
    """
    fig.update_traces(hovertemplate=template)
    return fig
