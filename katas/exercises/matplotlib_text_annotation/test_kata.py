"""Tests for matplotlib text annotation kata."""

import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
try:
    from user_kata import add_text_annotation
except ImportError:
    from .reference import add_text_annotation


def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


def test_adds_text_to_axes():
    """Test that text is added to the axes."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    texts = ax.texts
    assert len(texts) > 0, "Text annotation should be added"


def test_text_content_matches():
    """Test that the text content is correct."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Hello World")

    text_obj = ax.texts[0]
    assert text_obj.get_text() == "Hello World", "Text content should match"


def test_has_correct_fontsize():
    """Test that fontsize is 12."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    text_obj = ax.texts[0]
    assert text_obj.get_fontsize() == 12, "Font size should be 12"


def test_has_bbox():
    """Test that text has a bbox (background box)."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    text_obj = ax.texts[0]
    bbox = text_obj.get_bbox_patch()
    assert bbox is not None, "Text should have bbox"


def test_bbox_has_transparency():
    """Test that bbox has alpha=0.5."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    text_obj = ax.texts[0]
    bbox = text_obj.get_bbox_patch()
    assert bbox.get_facecolor()[3] == 0.5, "Bbox should have alpha=0.5"


def test_horizontal_alignment_center():
    """Test that text is center-aligned horizontally."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    text_obj = ax.texts[0]
    assert text_obj.get_horizontalalignment() == "center", "Text should be center-aligned"


def test_vertical_alignment_bottom():
    """Test that text is bottom-aligned vertically."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.5, 0.5, "Test")

    text_obj = ax.texts[0]
    assert text_obj.get_verticalalignment() == "bottom", "Text should be bottom-aligned"


def test_position_is_correct():
    """Test that text is positioned at the correct coordinates."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.3, 0.7, "Test")

    text_obj = ax.texts[0]
    pos = text_obj.get_position()
    assert pos[0] == 0.3, "X position should be correct"
    assert pos[1] == 0.7, "Y position should be correct"


def test_multiple_annotations():
    """Test that multiple annotations can be added."""

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_text_annotation(ax, 0.2, 0.2, "First")
    add_text_annotation(ax, 0.8, 0.8, "Second")

    assert len(ax.texts) == 2, "Should handle multiple annotations"
    assert ax.texts[0].get_text() == "First", "First annotation text should match"
    assert ax.texts[1].get_text() == "Second", "Second annotation text should match"
