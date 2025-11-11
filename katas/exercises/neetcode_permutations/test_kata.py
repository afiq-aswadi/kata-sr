"""Tests for Permutations kata."""

try:
    from user_kata import permute
except ImportError:
    from .reference import permute


def test_permute_example1():
    result = permute([1,2,3])
    result = [sorted(perm) for perm in result]
    result = sorted(result)
    expected = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    expected = [sorted(perm) for perm in expected]
    expected = sorted(expected)
    assert result == expected

def test_permute_example2():
    result = permute([0,1])
    result = sorted([sorted(perm) for perm in result])
    expected = [[0,1],[1,0]]
    expected = sorted([sorted(perm) for perm in expected])
    assert result == expected

def test_permute_example3():
    assert permute([1]) == [[1]]

def test_permute_two_elements():
    result = permute([1,2])
    result = sorted([sorted(perm) for perm in result])
    expected = [[1,2],[2,1]]
    expected = sorted([sorted(perm) for perm in expected])
    assert result == expected
