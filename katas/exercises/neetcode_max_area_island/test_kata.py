"""Tests for Max Area of Island kata."""

try:
    from user_kata import max_area_of_island
except ImportError:
    from .reference import max_area_of_island


def test_max_area_of_island_example1():
    grid = [
        [0,0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,1,1,0,1,0,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,0,0,1,0,1,0,0],
        [0,1,0,0,1,1,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,0,0,0]
    ]
    assert max_area_of_island(grid) == 6

def test_max_area_of_island_example2():
    grid = [[0,0,0,0,0,0,0,0]]
    assert max_area_of_island(grid) == 0

def test_max_area_of_island_single():
    grid = [[1]]
    assert max_area_of_island(grid) == 1

def test_max_area_of_island_multiple():
    grid = [[1,1,0,0],[1,0,0,1],[0,0,1,1]]
    assert max_area_of_island(grid) == 3
