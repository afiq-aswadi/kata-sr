"""Tests for Climbing Stairs kata."""

try:
    from user_kata import climb_stairs
except ImportError:
    from .reference import climb_stairs


def test_climb_stairs_example1():
    assert climb_stairs(2) == 2

def test_climb_stairs_example2():
    assert climb_stairs(3) == 3

def test_climb_stairs_base_case():
    assert climb_stairs(1) == 1

def test_climb_stairs_larger():
    assert climb_stairs(5) == 8
    assert climb_stairs(10) == 89
