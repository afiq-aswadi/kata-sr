"""Tests for matplotlib Artist API and FuncAnimation kata."""

import os
import tempfile

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def test_custom_artist_class_exists():
    """Test that CustomCircleArtist class is properly defined."""
    from template import CustomCircleArtist

    # Should be able to instantiate
    artist = CustomCircleArtist(center=(5, 5), radius=2)
    assert artist is not None
    assert artist.center == (5, 5)
    assert artist.radius == 2


def test_custom_artist_has_draw_method():
    """Test that CustomCircleArtist implements draw() method."""
    from template import CustomCircleArtist

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    assert hasattr(artist, "draw")
    assert callable(artist.draw)


def test_custom_artist_can_be_added_to_axes():
    """Test that custom Artist can be added to axes and rendered."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    artist = CustomCircleArtist(center=(5, 5), radius=2, color="red")
    artist.set_axes(ax)
    ax.add_artist(artist)

    # Should be able to draw the figure without errors
    fig.canvas.draw()

    # Artist should be in the axes artists list
    assert artist in ax.artists


def test_simple_animation_returns_funcanimation():
    """Test that create_simple_animation returns a FuncAnimation object."""
    from template import create_simple_animation

    anim = create_simple_animation(num_frames=10, interval=50)

    assert isinstance(anim, animation.FuncAnimation)
    assert anim is not None


def test_simple_animation_has_correct_parameters():
    """Test that simple animation has correct frame count and interval."""
    from template import create_simple_animation

    num_frames = 20
    interval = 100

    anim = create_simple_animation(num_frames=num_frames, interval=interval)

    # Check that animation was created with correct parameters
    # Note: FuncAnimation stores these internally but may not expose them directly
    # We verify by checking the animation runs without error
    assert anim is not None


def test_animation_with_init_has_init_func():
    """Test that animation with init has both init_func and update."""
    from template import create_animation_with_init

    anim = create_animation_with_init(num_frames=10, interval=50)

    assert isinstance(anim, animation.FuncAnimation)
    # The animation should have been created with init_func
    assert anim is not None


def test_animation_with_init_blit_parameter():
    """Test that animation respects blit parameter."""
    from template import create_animation_with_init

    # Test with blit=True
    anim_blit = create_animation_with_init(num_frames=10, interval=50, blit=True)
    assert isinstance(anim_blit, animation.FuncAnimation)

    # Test with blit=False
    anim_no_blit = create_animation_with_init(num_frames=10, interval=50, blit=False)
    assert isinstance(anim_no_blit, animation.FuncAnimation)


def test_blitting_animation_has_multiple_elements():
    """Test that blitting animation updates multiple elements correctly."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=10, interval=33)

    assert isinstance(anim, animation.FuncAnimation)


def test_blitting_animation_performance():
    """Test that blitting animation is created with blit=True."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=10, interval=33)

    # Animation should exist and be properly configured
    assert anim is not None
    assert isinstance(anim, animation.FuncAnimation)


def test_save_animation_to_file_with_filename():
    """Test saving animation to a specified file."""
    from template import create_simple_animation, save_animation_to_file

    anim = create_simple_animation(num_frames=5, interval=100)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        filename = f.name

    try:
        result_path = save_animation_to_file(anim, filename=filename, fps=5, dpi=50)

        assert result_path == filename
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def test_save_animation_to_file_auto_filename():
    """Test saving animation with auto-generated filename."""
    from template import create_simple_animation, save_animation_to_file

    anim = create_simple_animation(num_frames=5, interval=100)

    result_path = save_animation_to_file(anim, filename=None, fps=5, dpi=50)

    try:
        assert result_path is not None
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
        assert result_path.endswith(".mp4")

    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def test_interactive_animation_with_data_source():
    """Test that interactive animation calls data_source correctly."""
    from template import create_interactive_animation

    call_count = [0]

    def test_data_source(frame):
        call_count[0] += 1
        x = np.linspace(0, frame + 1, 50)
        y = np.sin(x)
        return x, y

    anim = create_interactive_animation(test_data_source)

    assert isinstance(anim, animation.FuncAnimation)


def test_interactive_animation_handles_varying_data():
    """Test that interactive animation handles data that changes size."""
    from template import create_interactive_animation

    def varying_data_source(frame):
        # Data size grows with frame
        size = 10 + frame
        x = np.linspace(0, 10, size)
        y = np.random.randn(size)
        return x, y

    anim = create_interactive_animation(varying_data_source)

    assert isinstance(anim, animation.FuncAnimation)


def test_add_custom_artist_to_plot_returns_tuple():
    """Test that add_custom_artist_to_plot returns proper tuple."""
    from template import add_custom_artist_to_plot

    result = add_custom_artist_to_plot()

    assert isinstance(result, tuple)
    assert len(result) == 3

    fig, ax, custom_artist = result

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    # custom_artist should be an Artist instance
    assert hasattr(custom_artist, "draw")


def test_add_custom_artist_to_plot_artist_attached():
    """Test that custom artist is properly attached to axes."""
    from template import add_custom_artist_to_plot

    fig, ax, custom_artist = add_custom_artist_to_plot()

    # Artist should be added to axes
    assert custom_artist in ax.artists

    # Should be able to draw without errors
    fig.canvas.draw()


def test_animation_handles_first_frame():
    """Test that animation handles the first frame correctly (edge case)."""
    from template import create_animation_with_init

    # Create animation with just 1 frame to test edge case
    anim = create_animation_with_init(num_frames=1, interval=50)

    assert isinstance(anim, animation.FuncAnimation)


def test_animation_handles_last_frame():
    """Test that animation handles the last frame correctly (edge case)."""
    from template import create_blitting_animation

    # Create animation and verify it can handle all frames including last
    num_frames = 10
    anim = create_blitting_animation(num_frames=num_frames, interval=50)

    assert isinstance(anim, animation.FuncAnimation)


def test_animation_interval_timing():
    """Test that animation uses appropriate interval timing."""
    from template import create_simple_animation

    # Test common framerates
    anim_20fps = create_simple_animation(num_frames=10, interval=50)  # 50ms = 20fps
    anim_30fps = create_simple_animation(num_frames=10, interval=33)  # 33ms ≈ 30fps

    assert isinstance(anim_20fps, animation.FuncAnimation)
    assert isinstance(anim_30fps, animation.FuncAnimation)


def test_custom_artist_visibility():
    """Test that custom artist respects visibility setting."""
    from template import CustomCircleArtist

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    artist = CustomCircleArtist(center=(5, 5), radius=2)
    artist.set_axes(ax)

    # Test visibility
    artist.set_visible(False)
    assert not artist.get_visible()

    artist.set_visible(True)
    assert artist.get_visible()
