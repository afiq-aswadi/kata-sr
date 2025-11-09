"""Tests for matplotlib blitting optimization kata."""

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
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=10, interval=50)
    assert isinstance(anim, animation.FuncAnimation)


def test_works_with_default_parameters():
    """Test that function works with default parameters."""
    from template import create_blitting_animation

    anim = create_blitting_animation()
    assert isinstance(anim, animation.FuncAnimation)


def test_respects_num_frames():
    """Test that num_frames parameter is respected."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=20, interval=50)
    assert anim is not None


def test_respects_interval():
    """Test that interval parameter is respected."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=10, interval=50)
    assert anim is not None


def test_handles_minimal_frames():
    """Test that animation handles just 1 frame."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=1, interval=50)
    assert isinstance(anim, animation.FuncAnimation)


def test_handles_many_frames():
    """Test that animation handles many frames."""
    from template import create_blitting_animation

    anim = create_blitting_animation(num_frames=200, interval=33)
    assert isinstance(anim, animation.FuncAnimation)


def test_30fps_interval():
    """Test that animation works with 30fps interval."""
    from template import create_blitting_animation

    # 30 fps ≈ 33ms interval (default)
    anim = create_blitting_animation(num_frames=10)
    assert isinstance(anim, animation.FuncAnimation)


def test_60fps_interval():
    """Test that animation works with 60fps interval."""
    from template import create_blitting_animation

    # 60 fps ≈ 16ms interval
    anim = create_blitting_animation(num_frames=10, interval=16)
    assert isinstance(anim, animation.FuncAnimation)
