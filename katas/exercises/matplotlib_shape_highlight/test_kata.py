"""Tests for matplotlib shape highlight kata."""

import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


def test_circle_adds_patch():
    """Test that circle adds a patch to axes."""
    from template import add_circle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_circle_highlight(ax, 0.5, 0.5, 0.2)

    patches = ax.patches
    assert len(patches) > 0, "Circle patch should be added"


def test_circle_center_position():
    """Test that circle is positioned at correct center."""
    from template import add_circle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_circle_highlight(ax, 0.3, 0.7, 0.1)

    circle = ax.patches[0]
    center = circle.center
    assert np.isclose(center[0], 0.3, atol=0.01), "Circle center x should be correct"
    assert np.isclose(center[1], 0.7, atol=0.01), "Circle center y should be correct"


def test_circle_radius():
    """Test that circle has correct radius."""
    from template import add_circle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_circle_highlight(ax, 0.5, 0.5, 0.25)

    circle = ax.patches[0]
    assert np.isclose(circle.radius, 0.25, atol=0.01), "Circle radius should be correct"


def test_circle_edge_styling():
    """Test that circle has colored edge."""
    from template import add_circle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_circle_highlight(ax, 0.5, 0.5, 0.2)

    circle = ax.patches[0]
    edgecolor = circle.get_edgecolor()
    # Should have a visible edge (not white or transparent)
    assert (
        edgecolor[0] != 1.0 or edgecolor[1] != 1.0
    ), "Circle should have colored edge"


def test_circle_linewidth():
    """Test that circle has appropriate linewidth."""
    from template import add_circle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_circle_highlight(ax, 0.5, 0.5, 0.2)

    circle = ax.patches[0]
    assert circle.get_linewidth() > 1, "Circle should have visible linewidth"


def test_rectangle_adds_patch():
    """Test that rectangle adds a patch to axes."""
    from template import add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_rectangle_highlight(ax, 0.2, 0.2, 0.3, 0.4)

    patches = ax.patches
    assert len(patches) > 0, "Rectangle patch should be added"


def test_rectangle_position():
    """Test that rectangle is positioned correctly."""
    from template import add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_rectangle_highlight(ax, 0.1, 0.3, 0.2, 0.4)

    rect = ax.patches[0]
    x, y = rect.get_xy()
    assert np.isclose(x, 0.1, atol=0.01), "Rectangle x position should be correct"
    assert np.isclose(y, 0.3, atol=0.01), "Rectangle y position should be correct"


def test_rectangle_dimensions():
    """Test that rectangle has correct width and height."""
    from template import add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_rectangle_highlight(ax, 0.2, 0.2, 0.5, 0.3)

    rect = ax.patches[0]
    assert np.isclose(rect.get_width(), 0.5, atol=0.01), "Rectangle width should be correct"
    assert np.isclose(rect.get_height(), 0.3, atol=0.01), "Rectangle height should be correct"


def test_rectangle_has_transparency():
    """Test that rectangle has alpha transparency."""
    from template import add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    add_rectangle_highlight(ax, 0.2, 0.2, 0.3, 0.4)

    rect = ax.patches[0]
    assert rect.get_facecolor()[3] < 1.0, "Rectangle should have transparency (alpha < 1)"


def test_multiple_shapes():
    """Test that multiple shapes can be added."""
    from template import add_circle_highlight, add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    add_circle_highlight(ax, 2, 2, 0.5)
    add_rectangle_highlight(ax, 5, 5, 2, 3)
    add_circle_highlight(ax, 8, 8, 1.0)

    assert len(ax.patches) == 3, "Should handle multiple shapes"


def test_shapes_with_different_sizes():
    """Test shapes with various sizes."""
    from template import add_circle_highlight, add_rectangle_highlight

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Small shapes
    add_circle_highlight(ax, 2, 2, 0.1)
    add_rectangle_highlight(ax, 5, 5, 0.2, 0.2)

    # Large shapes
    add_circle_highlight(ax, 7, 7, 2.0)
    add_rectangle_highlight(ax, 1, 1, 3, 3)

    assert len(ax.patches) == 4, "Should handle different shape sizes"
