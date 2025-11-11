"""Tests for Rotting Oranges kata."""

try:
    from user_kata import oranges_rotting
except ImportError:
    from .reference import oranges_rotting


def test_oranges_rotting_example1():
    grid = [[2,1,1],[1,1,0],[0,1,1]]
    assert oranges_rotting(grid) == 4

def test_oranges_rotting_example2():
    grid = [[2,1,1],[0,1,1],[1,0,1]]
    assert oranges_rotting(grid) == -1

def test_oranges_rotting_example3():
    grid = [[0,2]]
    assert oranges_rotting(grid) == 0

def test_oranges_rotting_no_fresh():
    grid = [[2,2],[2,2]]
    assert oranges_rotting(grid) == 0

def test_oranges_rotting_single():
    grid = [[2,1]]
    assert oranges_rotting(grid) == 1
