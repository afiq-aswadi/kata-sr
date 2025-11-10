"""Tests for Target Sum kata."""

def test_find_target_sum_ways_example1():
    from template import find_target_sum_ways
    assert find_target_sum_ways([1,1,1,1,1], 3) == 5

def test_find_target_sum_ways_example2():
    from template import find_target_sum_ways
    assert find_target_sum_ways([1], 1) == 1

def test_find_target_sum_ways_zero():
    from template import find_target_sum_ways
    assert find_target_sum_ways([0,0,1], 1) == 4

def test_find_target_sum_ways_impossible():
    from template import find_target_sum_ways
    assert find_target_sum_ways([1,2,3], 7) == 0
