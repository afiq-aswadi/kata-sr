"""Tests for Largest Rectangle in Histogram kata."""

try:
    from user_kata import largest_rectangle_area
except ImportError:
    from .reference import largest_rectangle_area


def test_largest_rectangle_area_example1():
    assert largest_rectangle_area([2,1,5,6,2,3]) == 10

def test_largest_rectangle_area_example2():
    assert largest_rectangle_area([2,4]) == 4
