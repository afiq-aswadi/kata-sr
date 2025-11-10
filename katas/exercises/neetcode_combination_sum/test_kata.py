"""Tests for Combination Sum kata."""

def test_combination_sum_example1():
    from template import combination_sum
    result = combination_sum([2,3,6,7], 7)
    result = [sorted(combo) for combo in result]
    result = sorted(result)
    expected = [[2,2,3],[7]]
    expected = [sorted(combo) for combo in expected]
    expected = sorted(expected)
    assert result == expected

def test_combination_sum_example2():
    from template import combination_sum
    result = combination_sum([2,3,5], 8)
    result = [sorted(combo) for combo in result]
    result = sorted(result)
    expected = [[2,2,2,2],[2,3,3],[3,5]]
    expected = [sorted(combo) for combo in expected]
    expected = sorted(expected)
    assert result == expected

def test_combination_sum_example3():
    from template import combination_sum
    result = combination_sum([2], 1)
    assert result == []

def test_combination_sum_single_element():
    from template import combination_sum
    result = combination_sum([1], 2)
    assert result == [[1,1]]
