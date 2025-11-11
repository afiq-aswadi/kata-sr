"""Tests for K Closest Points to Origin kata."""

try:
    from user_kata import k_closest
except ImportError:
    from .reference import k_closest


def test_k_closest_example1():
    result = k_closest([[1,3],[-2,2]], 1)
    assert result == [[-2,2]]

def test_k_closest_example2():
    result = k_closest([[3,3],[5,-1],[-2,4]], 2)
    # Order doesn't matter, so check as sets
    assert set(map(tuple, result)) == {(3,3), (-2,4)}

def test_k_closest_all_points():
    points = [[1,1],[2,2],[3,3]]
    result = k_closest(points, 3)
    assert len(result) == 3
    assert set(map(tuple, result)) == {(1,1), (2,2), (3,3)}

def test_k_closest_origin():
    result = k_closest([[0,0],[1,1],[2,2]], 1)
    assert result == [[0,0]]
