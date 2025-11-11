"""Tests for Alien Dictionary kata."""

try:
    from user_kata import alien_order
except ImportError:
    from .reference import alien_order


def test_alien_order_example1():
    result = alien_order(["wrt","wrf","er","ett","rftt"])
    # Verify it's a valid topological sort
    assert len(result) == 5
    assert set(result) == set("wertf")
    # Verify ordering constraints
    pos = {c: i for i, c in enumerate(result)}
    assert pos["w"] < pos["e"]
    assert pos["t"] < pos["f"]
    assert pos["r"] < pos["t"]
    assert pos["e"] < pos["r"]

def test_alien_order_example2():
    result = alien_order(["z","x"])
    assert result == "zx"

def test_alien_order_example3():
    result = alien_order(["z","x","z"])
    assert result == ""

def test_alien_order_simple():
    result = alien_order(["abc","ab"])
    assert result == ""
