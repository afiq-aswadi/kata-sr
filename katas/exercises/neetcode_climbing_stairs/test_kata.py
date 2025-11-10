"""Tests for Climbing Stairs kata."""

def test_climb_stairs_example1():
    from template import climb_stairs
    assert climb_stairs(2) == 2

def test_climb_stairs_example2():
    from template import climb_stairs
    assert climb_stairs(3) == 3

def test_climb_stairs_base_case():
    from template import climb_stairs
    assert climb_stairs(1) == 1

def test_climb_stairs_larger():
    from template import climb_stairs
    assert climb_stairs(5) == 8
    assert climb_stairs(10) == 89
