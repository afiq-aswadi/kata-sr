"""Tests for Two Sum kata."""

def test_two_sum_example1():
    from template import two_sum
    assert two_sum([2,7,11,15], 9) == [0,1]

def test_two_sum_example2():
    from template import two_sum
    assert two_sum([3,2,4], 6) == [1,2]

def test_two_sum_negative():
    from template import two_sum
    assert two_sum([-1,-2,-3,-4,-5], -8) == [2,4]
