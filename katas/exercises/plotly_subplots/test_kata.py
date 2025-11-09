"""Tests for Plotly subplots kata."""

import pytest
import plotly.graph_objects as go


def test_create_basic_subplot_grid():
    """Test basic 2x2 subplot grid creation."""
    from template import create_basic_subplot_grid

    data = {
        "scatter": {"x": [1, 2, 3], "y": [4, 5, 6]},
        "bar": {"x": ["A", "B", "C"], "y": [10, 20, 30]},
        "line": {"x": [0, 1, 2], "y": [1, 4, 9]},
        "heatmap": {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }

    fig = create_basic_subplot_grid(data)

    # Check figure is returned
    assert isinstance(fig, go.Figure)

    # Check we have 4 traces (one per subplot)
    assert len(fig.data) == 4

    # Check trace types
    assert isinstance(fig.data[0], go.Scatter)  # Scatter
    assert isinstance(fig.data[1], go.Bar)  # Bar
    assert isinstance(fig.data[2], go.Scatter)  # Line (also Scatter type)
    assert isinstance(fig.data[3], go.Heatmap)  # Heatmap

    # Check scatter mode
    assert fig.data[0].mode == "markers"
    assert fig.data[2].mode == "lines"

    # Check data is correct
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [4, 5, 6]


def test_create_shared_axes_figure():
    """Test shared x-axes across subplots."""
    from template import create_shared_axes_figure

    time_series_data = [
        {"x": [1, 2, 3, 4], "y": [10, 20, 15, 25], "name": "Series 1"},
        {"x": [1, 2, 3, 4], "y": [5, 15, 10, 20], "name": "Series 2"},
        {"x": [1, 2, 3, 4], "y": [8, 12, 18, 14], "name": "Series 3"},
    ]

    fig = create_shared_axes_figure(time_series_data)

    # Check figure is returned
    assert isinstance(fig, go.Figure)

    # Check we have 3 traces
    assert len(fig.data) == 3

    # Check all are line plots
    for trace in fig.data:
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"

    # Check data matches input
    assert list(fig.data[0].x) == [1, 2, 3, 4]
    assert list(fig.data[0].y) == [10, 20, 15, 25]
    assert fig.data[0].name == "Series 1"

    # Check layout has shared x-axes configured
    # The layout should have multiple y-axes but shared x-axis domain
    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert "yaxis3" in fig.layout


def test_create_secondary_yaxis_subplot():
    """Test dual y-axis subplot creation."""
    from template import create_secondary_yaxis_subplot

    primary_data = {"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "name": "Temperature"}
    secondary_data = {"x": [1, 2, 3, 4], "y": [100, 200, 150, 250], "name": "Pressure"}

    fig = create_secondary_yaxis_subplot(primary_data, secondary_data)

    # Check figure is returned
    assert isinstance(fig, go.Figure)

    # Check we have 2 traces
    assert len(fig.data) == 2

    # Check both are scatter plots
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)

    # Check data matches
    assert list(fig.data[0].x) == [1, 2, 3, 4]
    assert list(fig.data[0].y) == [10, 20, 30, 40]
    assert fig.data[0].name == "Temperature"

    assert list(fig.data[1].x) == [1, 2, 3, 4]
    assert list(fig.data[1].y) == [100, 200, 150, 250]
    assert fig.data[1].name == "Pressure"

    # Check that we have yaxis and yaxis2 (secondary)
    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout

    # Check y-axis titles
    assert fig.layout.yaxis.title.text == "Temperature"
    assert fig.layout.yaxis2.title.text == "Pressure"


def test_add_subplot_annotation():
    """Test adding annotation to specific subplot."""
    from template import add_subplot_annotation, create_basic_subplot_grid

    # Create a basic figure first
    data = {
        "scatter": {"x": [1, 2, 3], "y": [4, 5, 6]},
        "bar": {"x": ["A", "B", "C"], "y": [10, 20, 30]},
        "line": {"x": [0, 1, 2], "y": [1, 4, 9]},
        "heatmap": {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    fig = create_basic_subplot_grid(data)

    # Add annotation to subplot (1, 1)
    fig = add_subplot_annotation(
        fig, text="Peak", row=1, col=1, x=2, y=5
    )

    # Check annotation was added
    assert len(fig.layout.annotations) > 0

    # Find our annotation (filter out subplot titles if present)
    user_annotations = [
        ann for ann in fig.layout.annotations if ann.text == "Peak"
    ]
    assert len(user_annotations) == 1

    annotation = user_annotations[0]
    assert annotation.text == "Peak"
    assert annotation.x == 2
    assert annotation.y == 5
    assert annotation.showarrow is True


def test_create_complex_multipanel():
    """Test complex multi-panel figure with multiple features."""
    from template import create_complex_multipanel

    scatter_data = [
        {"x": [1, 2, 3, 4], "y": [10, 15, 13, 17], "name": "Primary"},
        {"x": [1, 2, 3, 4], "y": [100, 150, 130, 170], "name": "Secondary"},
    ]
    bar_data = {"x": ["Q1", "Q2", "Q3", "Q4"], "y": [25, 30, 28, 35]}
    heatmap_z = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    fig = create_complex_multipanel(scatter_data, bar_data, heatmap_z)

    # Check figure is returned
    assert isinstance(fig, go.Figure)

    # Check we have correct number of traces (2 scatter + 1 bar + 1 heatmap = 4)
    assert len(fig.data) == 4

    # Check trace types
    assert isinstance(fig.data[0], go.Scatter)
    assert isinstance(fig.data[1], go.Scatter)
    assert isinstance(fig.data[2], go.Bar)
    assert isinstance(fig.data[3], go.Heatmap)

    # Check scatter data
    assert list(fig.data[0].x) == [1, 2, 3, 4]
    assert list(fig.data[0].y) == [10, 15, 13, 17]
    assert fig.data[0].name == "Primary"

    assert list(fig.data[1].x) == [1, 2, 3, 4]
    assert list(fig.data[1].y) == [100, 150, 130, 170]
    assert fig.data[1].name == "Secondary"

    # Check bar data
    assert list(fig.data[2].x) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(fig.data[2].y) == [25, 30, 28, 35]

    # Check heatmap data (plotly returns tuples, not lists)
    assert list(map(list, fig.data[3].z)) == heatmap_z

    # Check layout has title
    assert fig.layout.title.text is not None
    assert len(fig.layout.title.text) > 0

    # Check we have multiple y-axes (at least yaxis and yaxis2 for dual y)
    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout


def test_update_individual_subplot_axes():
    """Test updating individual subplot axes."""
    from template import create_basic_subplot_grid, update_individual_subplot_axes

    # Create a basic figure first
    data = {
        "scatter": {"x": [1, 2, 3], "y": [4, 5, 6]},
        "bar": {"x": ["A", "B", "C"], "y": [10, 20, 30]},
        "line": {"x": [0, 1, 2], "y": [1, 4, 9]},
        "heatmap": {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    fig = create_basic_subplot_grid(data)

    # Define axis configurations
    axis_configs = [
        {"row": 1, "col": 1, "axis": "x", "title": "Time (s)", "range": [0, 5]},
        {"row": 1, "col": 1, "axis": "y", "title": "Value", "range": [0, 10]},
        {"row": 1, "col": 2, "axis": "y", "title": "Count"},
    ]

    fig = update_individual_subplot_axes(fig, axis_configs)

    # Check x-axis title for subplot (1,1)
    assert fig.layout.xaxis.title.text == "Time (s)"
    assert list(fig.layout.xaxis.range) == [0, 5]

    # Check y-axis title for subplot (1,1)
    assert fig.layout.yaxis.title.text == "Value"
    assert list(fig.layout.yaxis.range) == [0, 10]

    # Check y-axis title for subplot (1,2)
    assert fig.layout.yaxis2.title.text == "Count"


def test_subplot_grid_dimensions():
    """Test that subplot grid has correct dimensions."""
    from template import create_basic_subplot_grid

    data = {
        "scatter": {"x": [1, 2, 3], "y": [4, 5, 6]},
        "bar": {"x": ["A", "B", "C"], "y": [10, 20, 30]},
        "line": {"x": [0, 1, 2], "y": [1, 4, 9]},
        "heatmap": {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }

    fig = create_basic_subplot_grid(data)

    # Check that we have the expected axes (2x2 grid = 4 axes)
    # xaxis, xaxis2, xaxis3, xaxis4
    # yaxis, yaxis2, yaxis3, yaxis4
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout
    assert "xaxis3" in fig.layout
    assert "xaxis4" in fig.layout

    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert "yaxis3" in fig.layout
    assert "yaxis4" in fig.layout


def test_shared_axes_zooming_behavior():
    """Test that shared axes are properly configured."""
    from template import create_shared_axes_figure

    time_series_data = [
        {"x": [1, 2, 3], "y": [10, 20, 15], "name": "Series 1"},
        {"x": [1, 2, 3], "y": [5, 15, 10], "name": "Series 2"},
        {"x": [1, 2, 3], "y": [8, 12, 18], "name": "Series 3"},
    ]

    fig = create_shared_axes_figure(time_series_data)

    # Check that xaxis domains are properly configured
    # In shared x-axes configuration, all x-axes should share the same domain
    # or be linked via the 'matches' attribute or 'anchor' configuration
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout
    assert "xaxis3" in fig.layout

    # Check for shared axes configuration
    # Plotly can represent this in different ways:
    # 1. via 'matches' attribute (can match to any other x-axis, not just x)
    # 2. via having the same domain
    # 3. via anchor relationships
    has_shared_config = False

    # Check for matches attribute - any x-axis matching another x-axis indicates shared axes
    for i in range(1, 4):  # Check xaxis, xaxis2 and xaxis3
        if i == 1:
            axis_name = "xaxis"
        else:
            axis_name = f"xaxis{i}"

        if axis_name in fig.layout:
            axis = fig.layout[axis_name]
            # Check if matches is set to any x-axis (x, x1, x2, x3, etc.)
            if hasattr(axis, "matches") and axis.matches is not None and axis.matches != '':
                has_shared_config = True
                break

    assert has_shared_config, "Shared x-axes should be configured"
