"""Tests for Two Sum kata."""

try:
    from user_kata import two_sum
except ImportError:
    from .reference import two_sum


def test_two_sum_example1():
    assert two_sum([2,7,11,15], 9) == [0,1]

def test_two_sum_example2():
    assert two_sum([3,2,4], 6) == [1,2]

def test_two_sum_negative():
    assert two_sum([-1,-2,-3,-4,-5], -8) == [2,4]
