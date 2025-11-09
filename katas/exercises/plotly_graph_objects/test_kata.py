"""Tests for Plotly graph_objects basics kata."""

import pytest
import plotly.graph_objects as go


def test_create_scatter_plot():
    from template import create_scatter_plot

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    title = "Test Scatter"

    fig = create_scatter_plot(x, y, title)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y
    assert fig.data[0].marker.size == 10
    assert fig.data[0].marker.color == "blue"
    assert fig.layout.title.text == title


def test_create_line_plot():
    from template import create_line_plot

    x = [1, 2, 3, 4]
    y = [10, 20, 15, 25]
    name = "Test Line"
    color = "green"

    fig = create_line_plot(x, y, name, color)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y
    assert fig.data[0].name == name
    assert fig.data[0].line.color == color


def test_create_bar_chart():
    from template import create_bar_chart

    categories = ["A", "B", "C", "D"]
    values = [10, 25, 15, 30]
    title = "Test Bar Chart"

    fig = create_bar_chart(categories, values, title)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].x) == categories
    assert list(fig.data[0].y) == values
    assert fig.layout.title.text == title


def test_add_custom_hover_template():
    from template import add_custom_hover_template

    # Create a simple figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

    template = "Value: %{y}<extra></extra>"
    result_fig = add_custom_hover_template(fig, template)

    assert isinstance(result_fig, go.Figure)
    assert result_fig.data[0].hovertemplate == template


def test_create_multi_trace_plot():
    from template import create_multi_trace_plot

    x = [1, 2, 3, 4]
    y1 = [10, 12, 15, 14]
    y2 = [10, 11.5, 13, 14.5]

    fig = create_multi_trace_plot(x, y1, y2)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2

    # Check first trace (scatter)
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"
    assert fig.data[0].name == "Data Points"
    assert fig.data[0].marker.color == "blue"

    # Check second trace (line)
    assert fig.data[1].type == "scatter"
    assert fig.data[1].mode == "lines"
    assert fig.data[1].name == "Trend"
    assert fig.data[1].line.color == "red"
    assert fig.data[1].line.dash == "dash"


def test_configure_layout():
    from template import configure_layout

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

    title = "Main Title"
    x_label = "X Axis"
    y_label = "Y Axis"

    result_fig = configure_layout(fig, title, x_label, y_label, showlegend=True)

    assert isinstance(result_fig, go.Figure)
    assert result_fig.layout.title.text == title
    assert result_fig.layout.xaxis.title.text == x_label
    assert result_fig.layout.yaxis.title.text == y_label
    assert result_fig.layout.showlegend is True


def test_configure_layout_no_legend():
    from template import configure_layout

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

    result_fig = configure_layout(fig, "Title", "X", "Y", showlegend=False)

    assert result_fig.layout.showlegend is False


def test_create_styled_scatter_with_colors():
    from template import create_styled_scatter

    x = [1, 2, 3, 4]
    y = [10, 20, 15, 25]
    colors = ["red", "blue", "green", "yellow"]

    fig = create_styled_scatter(x, y, colors=colors)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].marker.color) == colors


def test_create_styled_scatter_with_sizes():
    from template import create_styled_scatter

    x = [1, 2, 3, 4]
    y = [10, 20, 15, 25]
    sizes = [5, 10, 15, 20]

    fig = create_styled_scatter(x, y, sizes=sizes)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].marker.size) == sizes


def test_create_styled_scatter_with_both():
    from template import create_styled_scatter

    x = [1, 2, 3]
    y = [10, 20, 15]
    colors = ["red", "blue", "green"]
    sizes = [8, 12, 16]

    fig = create_styled_scatter(x, y, colors=colors, sizes=sizes)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].marker.color) == colors
    assert list(fig.data[0].marker.size) == sizes


def test_add_hover_formatting():
    from template import add_hover_formatting

    x = [1.0, 2.0, 3.0]
    y = [10.5, 20.75, 15.333]
    names = ["Point A", "Point B", "Point C"]

    fig = add_hover_formatting(x, y, names)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].text) == names
    assert fig.data[0].hovertemplate is not None
    # Check that template includes the expected format
    assert "%{text}" in fig.data[0].hovertemplate
    assert "%{x}" in fig.data[0].hovertemplate
    assert "%{y:.2f}" in fig.data[0].hovertemplate


def test_create_grouped_bar_chart():
    from template import create_grouped_bar_chart

    categories = ["Q1", "Q2", "Q3", "Q4"]
    group1 = [10, 15, 13, 17]
    group2 = [12, 14, 16, 15]

    fig = create_grouped_bar_chart(categories, group1, group2)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2

    # Check first bar group
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].x) == categories
    assert list(fig.data[0].y) == group1

    # Check second bar group
    assert fig.data[1].type == "bar"
    assert list(fig.data[1].x) == categories
    assert list(fig.data[1].y) == group2


def test_create_grouped_bar_chart_with_names():
    from template import create_grouped_bar_chart

    categories = ["A", "B", "C"]
    group1 = [10, 20, 30]
    group2 = [15, 25, 35]

    fig = create_grouped_bar_chart(
        categories, group1, group2, group1_name="Team A", group2_name="Team B"
    )

    assert fig.data[0].name == "Team A"
    assert fig.data[1].name == "Team B"


def test_scatter_plot_data_integrity():
    from template import create_scatter_plot

    x = [1.5, 2.5, 3.5]
    y = [100.25, 200.75, 150.5]

    fig = create_scatter_plot(x, y, "Data Integrity Test")

    # Ensure data is preserved exactly
    assert list(fig.data[0].x) == x
    assert list(fig.data[0].y) == y


def test_line_plot_default_color():
    from template import create_line_plot

    x = [1, 2, 3]
    y = [4, 5, 6]

    fig = create_line_plot(x, y, "Test Line")

    # Should use default red color
    assert fig.data[0].line.color == "red"
