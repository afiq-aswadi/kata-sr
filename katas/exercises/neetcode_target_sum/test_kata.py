"""Tests for Target Sum kata."""

try:
    from user_kata import find_target_sum_ways
except ImportError:
    from .reference import find_target_sum_ways


def test_find_target_sum_ways_example1():
    assert find_target_sum_ways([1,1,1,1,1], 3) == 5

def test_find_target_sum_ways_example2():
    assert find_target_sum_ways([1], 1) == 1

def test_find_target_sum_ways_zero():
    assert find_target_sum_ways([0,0,1], 1) == 4

def test_find_target_sum_ways_impossible():
    assert find_target_sum_ways([1,2,3], 7) == 0
