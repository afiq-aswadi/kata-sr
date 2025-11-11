"""Tests for Spiral Matrix kata."""

try:
    from user_kata import spiral_order
except ImportError:
    from .reference import spiral_order


def test_spiral_order_example1():
    assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]

def test_spiral_order_example2():
    assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]

def test_spiral_order_single():
    assert spiral_order([[1]]) == [1]

def test_spiral_order_single_row():
    assert spiral_order([[1,2,3,4]]) == [1,2,3,4]

def test_spiral_order_single_col():
    assert spiral_order([[1],[2],[3]]) == [1,2,3]
