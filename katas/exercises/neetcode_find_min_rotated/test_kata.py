"""Tests for Find Minimum in Rotated Sorted Array kata."""

try:
    from user_kata import find_min
except ImportError:
    from .reference import find_min


def test_find_min_example1():
    assert find_min([3,4,5,1,2]) == 1

def test_find_min_example2():
    assert find_min([4,5,6,7,0,1,2]) == 0

def test_find_min_no_rotation():
    assert find_min([1,2,3,4,5]) == 1
