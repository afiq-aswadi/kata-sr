"""Tests for Largest Rectangle in Histogram kata."""

def test_largest_rectangle_area_example1():
    from template import largest_rectangle_area
    assert largest_rectangle_area([2,1,5,6,2,3]) == 10

def test_largest_rectangle_area_example2():
    from template import largest_rectangle_area
    assert largest_rectangle_area([2,4]) == 4
