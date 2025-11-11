"""Tests for 3Sum kata."""

try:
    from user_kata import three_sum
except ImportError:
    from .reference import three_sum


def test_three_sum_example1():
    result = three_sum([-1,0,1,2,-1,-4])
    result = [sorted(triplet) for triplet in result]
    result = sorted(result)
    expected = [[-1,-1,2],[-1,0,1]]
    expected = [sorted(triplet) for triplet in expected]
    expected = sorted(expected)
    assert result == expected

def test_three_sum_example2():
    assert three_sum([0,1,1]) == []

def test_three_sum_example3():
    assert three_sum([0,0,0]) == [[0,0,0]]
