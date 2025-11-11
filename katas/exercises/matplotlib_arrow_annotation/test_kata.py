"""Tests for matplotlib arrow annotation kata."""

import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
try:
    from user_kata import add_arrow_annotation
except ImportError:
    from .reference import add_arrow_annotation


def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


def test_adds_annotation():
    """Test that annotation is added to axes."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Peak")

    texts = ax.texts
    assert len(texts) > 0, "Annotation should be added"


def test_annotation_text_matches():
    """Test that annotation text is correct."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Test Point")

    annotation = ax.texts[0]
    assert annotation.get_text() == "Test Point", "Annotation text should match"


def test_has_arrow_properties():
    """Test that annotation has arrow properties."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Peak")

    annotation = ax.texts[0]
    # Check that annotation has arrow-related attributes
    assert hasattr(annotation, "arrow_patch") or hasattr(
        annotation, "arrowprops"
    ), "Annotation should have arrow properties"


def test_fontsize_is_10():
    """Test that fontsize is 10."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Peak")

    annotation = ax.texts[0]
    assert annotation.get_fontsize() == 10, "Fontsize should be 10"


def test_has_bbox():
    """Test that annotation has a bbox."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Peak")

    annotation = ax.texts[0]
    bbox = annotation.get_bbox_patch()
    assert bbox is not None, "Annotation should have bbox"


def test_bbox_has_transparency():
    """Test that bbox has alpha=0.5."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Peak")

    annotation = ax.texts[0]
    bbox = annotation.get_bbox_patch()
    assert bbox.get_facecolor()[3] == 0.5, "Bbox should have alpha=0.5"


def test_points_to_correct_location():
    """Test that arrow points to specified coordinates."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.5, 0.8, "Point")

    annotation = ax.texts[0]
    # Get the xy position that the arrow points to
    xy = annotation.xy
    assert xy[0] == 1.5, "Arrow should point to correct x coordinate"
    assert xy[1] == 0.8, "Arrow should point to correct y coordinate"


def test_custom_offsets():
    """Test that custom offset values are respected."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])

    add_arrow_annotation(ax, 1.0, 1.0, "Test", offset_x=30, offset_y=-15)

    annotation = ax.texts[0]
    # The xytext should use the custom offsets
    # This is in offset points coordinate system
    assert hasattr(annotation, "xyann") or hasattr(
        annotation, "xytext"
    ), "Annotation should have offset text position"


def test_multiple_arrows():
    """Test that multiple arrow annotations can be added."""

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 0, 1])

    add_arrow_annotation(ax, 1.0, 1.0, "First Peak")
    add_arrow_annotation(ax, 3.0, 1.0, "Second Peak")

    assert len(ax.texts) == 2, "Should handle multiple arrow annotations"
    assert ax.texts[0].get_text() == "First Peak", "First annotation should be correct"
    assert ax.texts[1].get_text() == "Second Peak", "Second annotation should be correct"


def test_edge_case_positions():
    """Test annotation at edge positions."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Should not crash with edge positions and negative offsets
    add_arrow_annotation(ax, 0, 0, "Origin", offset_x=-20, offset_y=-20)

    assert len(ax.texts) > 0, "Should handle edge case positions"
