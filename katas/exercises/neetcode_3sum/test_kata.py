"""Tests for 3Sum kata."""

def test_three_sum_example1():
    from template import three_sum
    result = three_sum([-1,0,1,2,-1,-4])
    result = [sorted(triplet) for triplet in result]
    result = sorted(result)
    expected = [[-1,-1,2],[-1,0,1]]
    expected = [sorted(triplet) for triplet in expected]
    expected = sorted(expected)
    assert result == expected

def test_three_sum_example2():
    from template import three_sum
    assert three_sum([0,1,1]) == []

def test_three_sum_example3():
    from template import three_sum
    assert three_sum([0,0,0]) == [[0,0,0]]
