"""Tests for Maximum Subarray kata."""

try:
    from user_kata import max_subarray
except ImportError:
    from .reference import max_subarray


def test_max_subarray_example1():
    assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6

def test_max_subarray_example2():
    assert max_subarray([1]) == 1

def test_max_subarray_example3():
    assert max_subarray([5,4,-1,7,8]) == 23

def test_max_subarray_all_negative():
    assert max_subarray([-2,-1]) == -1

def test_max_subarray_mixed():
    assert max_subarray([-2,1]) == 1
