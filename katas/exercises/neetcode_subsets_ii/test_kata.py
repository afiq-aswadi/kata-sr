"""Tests for Subsets II kata."""

def test_subsets_with_dup_example1():
    from template import subsets_with_dup
    result = subsets_with_dup([1,2,2])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1],[1,2],[1,2,2],[2],[2,2]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_with_dup_example2():
    from template import subsets_with_dup
    result = subsets_with_dup([0])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[0]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_with_dup_all_same():
    from template import subsets_with_dup
    result = subsets_with_dup([1,1,1])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1],[1,1],[1,1,1]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected
