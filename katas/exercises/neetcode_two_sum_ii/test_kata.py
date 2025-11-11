"""Tests for Two Sum II kata."""

try:
    from user_kata import two_sum_ii
except ImportError:
    from .reference import two_sum_ii


def test_two_sum_ii_example1():
    assert two_sum_ii([2,7,11,15], 9) == [1,2]

def test_two_sum_ii_example2():
    assert two_sum_ii([2,3,4], 6) == [1,3]

def test_two_sum_ii_example3():
    assert two_sum_ii([-1,0], -1) == [1,2]

def test_two_sum_ii_negative():
    assert two_sum_ii([-5,-3,0,2,4], -5) == [1,3]

def test_two_sum_ii_duplicates():
    assert two_sum_ii([1,2,3,4,4,9,56,90], 8) == [4,5]
