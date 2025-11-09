"""Tests for matplotlib custom Artist kata."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def test_custom_artist_class_exists():
    """Test that CustomCircleArtist class is defined."""
    from template import CustomCircleArtist

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    assert artist is not None
    assert artist.center == (5, 5)
    assert artist.radius == 2
    assert artist.color == "blue"


def test_custom_artist_has_draw_method():
    """Test that CustomCircleArtist implements draw() method."""
    from template import CustomCircleArtist

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    assert hasattr(artist, "draw")
    assert callable(artist.draw)


def test_custom_artist_inherits_from_artist():
    """Test that CustomCircleArtist inherits from matplotlib Artist."""
    import matplotlib.artist as mpl_artist

    from template import CustomCircleArtist

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    assert isinstance(artist, mpl_artist.Artist)


def test_custom_artist_can_be_added_to_axes():
    """Test that custom artist can be added to axes without errors."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    artist = CustomCircleArtist(center=(5, 5), radius=2, color="red")
    artist.set_axes(ax)
    ax.add_artist(artist)

    # Should be in the axes artists list
    assert artist in ax.artists


def test_custom_artist_renders_without_error():
    """Test that the artist can be rendered without raising exceptions."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    artist.set_axes(ax)
    ax.add_artist(artist)

    # Should draw without errors
    fig.canvas.draw()


def test_custom_artist_respects_visibility():
    """Test that artist respects visibility setting."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    artist.set_axes(ax)

    # Test visibility get/set
    artist.set_visible(False)
    assert not artist.get_visible()

    artist.set_visible(True)
    assert artist.get_visible()

    # Should still render without error when visible
    ax.add_artist(artist)
    fig.canvas.draw()


def test_custom_artist_with_different_colors():
    """Test that custom color parameter works."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    colors = ["red", "green", "blue", "purple"]
    for color in colors:
        artist = CustomCircleArtist(center=(5, 5), radius=2, color=color)
        assert artist.color == color
        artist.set_axes(ax)
        ax.add_artist(artist)

    # Should render all without errors
    fig.canvas.draw()


def test_custom_artist_with_different_positions():
    """Test that artist works at different positions."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    positions = [(5, 5), (10, 10), (15, 15), (2, 18)]
    for pos in positions:
        artist = CustomCircleArtist(center=pos, radius=1.5)
        artist.set_axes(ax)
        ax.add_artist(artist)

    # Should render all without errors
    fig.canvas.draw()


def test_custom_artist_with_different_radii():
    """Test that artist works with different radii."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    radii = [0.5, 1.0, 2.0, 3.0]
    for radius in radii:
        artist = CustomCircleArtist(center=(5, 5), radius=radius)
        assert artist.radius == radius
        artist.set_axes(ax)

    # Should work with all radii
    fig.canvas.draw()
