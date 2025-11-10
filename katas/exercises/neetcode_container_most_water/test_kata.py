"""Tests for Container With Most Water kata."""

def test_max_area_example1():
    from template import max_area
    assert max_area([1,8,6,2,5,4,8,3,7]) == 49

def test_max_area_example2():
    from template import max_area
    assert max_area([1,1]) == 1

def test_max_area_increasing():
    from template import max_area
    assert max_area([1,2,3,4,5]) == 6
