"""Tests for Kth Largest Element in an Array kata."""

try:
    from user_kata import find_kth_largest
except ImportError:
    from .reference import find_kth_largest


def test_kth_largest_example1():
    assert find_kth_largest([3,2,1,5,6,4], 2) == 5

def test_kth_largest_example2():
    assert find_kth_largest([3,2,3,1,2,4,5,5,6], 4) == 4

def test_kth_largest_first():
    assert find_kth_largest([1,2,3,4,5], 1) == 5

def test_kth_largest_last():
    assert find_kth_largest([5,4,3,2,1], 5) == 1

def test_kth_largest_duplicates():
    assert find_kth_largest([1,1,1,1,1], 3) == 1
