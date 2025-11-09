"""Tests for matplotlib simple animation kata."""

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def test_returns_funcanimation():
    """Test that function returns a FuncAnimation object."""
    from template import create_simple_animation

    anim = create_simple_animation(num_frames=10, interval=50)
    assert isinstance(anim, animation.FuncAnimation)


def test_creates_with_default_parameters():
    """Test that function works with default parameters."""
    from template import create_simple_animation

    anim = create_simple_animation()
    assert isinstance(anim, animation.FuncAnimation)


def test_respects_num_frames_parameter():
    """Test that num_frames parameter is used."""
    from template import create_simple_animation

    num_frames = 20
    anim = create_simple_animation(num_frames=num_frames)
    assert anim is not None


def test_respects_interval_parameter():
    """Test that interval parameter is used."""
    from template import create_simple_animation

    anim_50 = create_simple_animation(num_frames=10, interval=50)
    anim_100 = create_simple_animation(num_frames=10, interval=100)

    assert anim_50 is not None
    assert anim_100 is not None


def test_animation_with_minimal_frames():
    """Test that animation works with just 1 frame."""
    from template import create_simple_animation

    anim = create_simple_animation(num_frames=1, interval=50)
    assert isinstance(anim, animation.FuncAnimation)


def test_animation_with_many_frames():
    """Test that animation works with many frames."""
    from template import create_simple_animation

    anim = create_simple_animation(num_frames=200, interval=33)
    assert isinstance(anim, animation.FuncAnimation)


def test_common_frame_rates():
    """Test animation with common frame rate intervals."""
    from template import create_simple_animation

    # 20 fps = 50ms interval
    anim_20fps = create_simple_animation(num_frames=10, interval=50)
    assert isinstance(anim_20fps, animation.FuncAnimation)

    # 30 fps ≈ 33ms interval
    anim_30fps = create_simple_animation(num_frames=10, interval=33)
    assert isinstance(anim_30fps, animation.FuncAnimation)


def test_animation_creates_figure():
    """Test that animation creates and returns a figure."""
    from template import create_simple_animation

    anim = create_simple_animation(num_frames=5)

    # Animation should have reference to figure
    assert anim is not None
    assert isinstance(anim, animation.FuncAnimation)
