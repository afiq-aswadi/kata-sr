"""Tests for Find Minimum in Rotated Sorted Array kata."""

def test_find_min_example1():
    from template import find_min
    assert find_min([3,4,5,1,2]) == 1

def test_find_min_example2():
    from template import find_min
    assert find_min([4,5,6,7,0,1,2]) == 0

def test_find_min_no_rotation():
    from template import find_min
    assert find_min([1,2,3,4,5]) == 1
