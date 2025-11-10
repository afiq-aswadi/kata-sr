"""Tests for Subsets kata."""

def test_subsets_example1():
    from template import subsets
    result = subsets([1,2,3])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_example2():
    from template import subsets
    result = subsets([0])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[0]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected

def test_subsets_single():
    from template import subsets
    result = subsets([1])
    result = [sorted(subset) for subset in result]
    result = sorted(result)
    expected = [[],[1]]
    expected = [sorted(subset) for subset in expected]
    expected = sorted(expected)
    assert result == expected
