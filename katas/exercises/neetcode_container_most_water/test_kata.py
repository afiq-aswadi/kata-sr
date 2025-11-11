"""Tests for Container With Most Water kata."""

try:
    from user_kata import max_area
except ImportError:
    from .reference import max_area


def test_max_area_example1():
    assert max_area([1,8,6,2,5,4,8,3,7]) == 49

def test_max_area_example2():
    assert max_area([1,1]) == 1

def test_max_area_increasing():
    assert max_area([1,2,3,4,5]) == 6
