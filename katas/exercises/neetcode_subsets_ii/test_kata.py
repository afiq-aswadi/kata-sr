"""Tests for Subsets II kata."""

try:
    from user_kata import subsets_with_dup
except ImportError:
    from .reference import subsets_with_dup


def test_subsets_with_dup_example1():
    result = subsets_with_dup([1,2,2])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1],[1,2],[1,2,2],[2],[2,2]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_with_dup_example2():
    result = subsets_with_dup([0])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[0]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_with_dup_all_same():
    result = subsets_with_dup([1,1,1])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1],[1,1],[1,1,1]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected
