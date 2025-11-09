"""Tests for plotly_subplot_annotation kata."""

import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

try:
    from user_kata import add_annotation_to_subplot
except ImportError:
    from .reference import add_annotation_to_subplot


def test_returns_figure():
    """Function should return a Plotly Figure object."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Test", 1, 1, 0, 0)
    assert isinstance(result, go.Figure)


def test_annotation_added():
    """Should add annotation to the figure."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Peak", 1, 1, 2, 5)

    # Should have at least one annotation
    assert len(result.layout.annotations) > 0


def test_annotation_text():
    """Annotation should contain the specified text."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Maximum", 1, 1, 3, 7)

    # Find the annotation with our text
    matching_annotations = [a for a in result.layout.annotations if a.text == "Maximum"]
    assert len(matching_annotations) == 1


def test_annotation_coordinates():
    """Annotation should be at specified coordinates."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Point", 1, 1, 10, 20)

    # Find our annotation (filter out subplot titles)
    user_annotations = [a for a in result.layout.annotations if a.text == "Point"]
    assert len(user_annotations) == 1

    annotation = user_annotations[0]
    assert annotation.x == 10
    assert annotation.y == 20


def test_subplot_11_axis_reference():
    """Subplot (1,1) should use 'x' and 'y' axis references."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Test", 1, 1, 5, 10)

    user_annotations = [a for a in result.layout.annotations if a.text == "Test"]
    annotation = user_annotations[0]

    assert annotation.xref == "x"
    assert annotation.yref == "y"


def test_subplot_12_axis_reference():
    """Subplot (1,2) should use 'x2' and 'y2' axis references."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Test", 1, 2, 5, 10)

    user_annotations = [a for a in result.layout.annotations if a.text == "Test"]
    annotation = user_annotations[0]

    assert annotation.xref == "x2"
    assert annotation.yref == "y2"


def test_subplot_21_axis_reference():
    """Subplot (2,1) should use 'x3' and 'y3' axis references."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Test", 2, 1, 5, 10)

    user_annotations = [a for a in result.layout.annotations if a.text == "Test"]
    annotation = user_annotations[0]

    assert annotation.xref == "x3"
    assert annotation.yref == "y3"


def test_multiple_annotations():
    """Should be able to add multiple annotations to different subplots."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Ann1", 1, 1, 1, 1)
    result = add_annotation_to_subplot(result, "Ann2", 1, 2, 2, 2)
    result = add_annotation_to_subplot(result, "Ann3", 2, 1, 3, 3)

    # Should have 3 user annotations (plus subplot titles)
    user_annotations = [a for a in result.layout.annotations if a.text in ["Ann1", "Ann2", "Ann3"]]
    assert len(user_annotations) == 3


def test_arrow_enabled():
    """Annotation should have arrow enabled."""
    fig = make_subplots(rows=2, cols=2)
    result = add_annotation_to_subplot(fig, "Arrow", 1, 1, 5, 5)

    user_annotations = [a for a in result.layout.annotations if a.text == "Arrow"]
    annotation = user_annotations[0]

    assert annotation.showarrow is True
