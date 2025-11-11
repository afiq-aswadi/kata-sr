"""Tests for Combination Sum II kata."""

try:
    from user_kata import combination_sum2
except ImportError:
    from .reference import combination_sum2


def test_combination_sum2_example1():
    result = combination_sum2([10,1,2,7,6,1,5], 8)
    result = [sorted(combo) for combo in result]
    result = sorted(result)
    expected = [[1,1,6],[1,2,5],[1,7],[2,6]]
    expected = [sorted(combo) for combo in expected]
    expected = sorted(expected)
    assert result == expected

def test_combination_sum2_example2():
    result = combination_sum2([2,5,2,1,2], 5)
    result = [sorted(combo) for combo in result]
    result = sorted(result)
    expected = [[1,2,2],[5]]
    expected = [sorted(combo) for combo in expected]
    expected = sorted(expected)
    assert result == expected

def test_combination_sum2_no_solution():
    result = combination_sum2([2], 1)
    assert result == []

def test_combination_sum2_exact_match():
    result = combination_sum2([1,1,1], 2)
    result = [sorted(combo) for combo in result]
    result = sorted(result)
    expected = [[1,1]]
    expected = [sorted(combo) for combo in expected]
    expected = sorted(expected)
    assert result == expected
