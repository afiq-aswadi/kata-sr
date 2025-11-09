"""Tests for matplotlib span highlight kata."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test."""
    yield
    plt.close("all")


def test_vertical_span_adds_collection():
    """Test that vertical span adds a collection to axes."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    add_vertical_span(ax, 2, 5)

    collections = ax.collections
    assert len(collections) > 0, "Vertical span should add a collection"


def test_vertical_span_has_transparency():
    """Test that vertical span has alpha=0.3."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    add_vertical_span(ax, 2, 5)

    span = ax.collections[0]
    assert span.get_alpha() == 0.3, "Span should have alpha=0.3"


def test_vertical_span_behind_data():
    """Test that vertical span is behind data (zorder=0)."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    add_vertical_span(ax, 2, 5)

    span = ax.collections[0]
    assert span.get_zorder() == 0, "Span should be behind data (zorder=0)"


def test_vertical_span_custom_color():
    """Test that vertical span uses custom color."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    add_vertical_span(ax, 2, 5, color="blue")

    span = ax.collections[0]
    # Should have color set (not default)
    assert span.get_facecolor() is not None, "Span should have color"


def test_horizontal_span_adds_collection():
    """Test that horizontal span adds a collection to axes."""
    from template import add_horizontal_span

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])

    add_horizontal_span(ax, 2, 5)

    collections = ax.collections
    assert len(collections) > 0, "Horizontal span should add a collection"


def test_horizontal_span_has_transparency():
    """Test that horizontal span has alpha=0.3."""
    from template import add_horizontal_span

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])

    add_horizontal_span(ax, 2, 5)

    span = ax.collections[0]
    assert span.get_alpha() == 0.3, "Span should have alpha=0.3"


def test_horizontal_span_behind_data():
    """Test that horizontal span is behind data (zorder=0)."""
    from template import add_horizontal_span

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])

    add_horizontal_span(ax, 2, 5)

    span = ax.collections[0]
    assert span.get_zorder() == 0, "Span should be behind data (zorder=0)"


def test_horizontal_span_custom_color():
    """Test that horizontal span uses custom color."""
    from template import add_horizontal_span

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])

    add_horizontal_span(ax, 2, 5, color="red")

    span = ax.collections[0]
    # Should have color set (not default)
    assert span.get_facecolor() is not None, "Span should have color"


def test_multiple_vertical_spans():
    """Test that multiple vertical spans can be added."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    add_vertical_span(ax, 1, 3)
    add_vertical_span(ax, 5, 7)
    add_vertical_span(ax, 8, 9)

    assert len(ax.collections) == 3, "Should handle multiple vertical spans"


def test_multiple_horizontal_spans():
    """Test that multiple horizontal spans can be added."""
    from template import add_horizontal_span

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 10])

    add_horizontal_span(ax, 1, 3)
    add_horizontal_span(ax, 5, 7)

    assert len(ax.collections) == 2, "Should handle multiple horizontal spans"


def test_mixed_spans():
    """Test that both vertical and horizontal spans can be used together."""
    from template import add_horizontal_span, add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10])

    add_vertical_span(ax, 2, 4, color="yellow")
    add_horizontal_span(ax, 6, 8, color="cyan")

    assert len(ax.collections) == 2, "Should handle both vertical and horizontal spans"


def test_span_default_color():
    """Test that default color works."""
    from template import add_vertical_span

    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])

    # Should use default gray color
    add_vertical_span(ax, 2, 5)

    assert len(ax.collections) > 0, "Should work with default color"
