"""Tests for Max Area of Island kata."""

def test_max_area_of_island_example1():
    from template import max_area_of_island
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
    from template import max_area_of_island
    grid = [[0,0,0,0,0,0,0,0]]
    assert max_area_of_island(grid) == 0

def test_max_area_of_island_single():
    from template import max_area_of_island
    grid = [[1]]
    assert max_area_of_island(grid) == 1

def test_max_area_of_island_multiple():
    from template import max_area_of_island
    grid = [[1,1,0,0],[1,0,0,1],[0,0,1,1]]
    assert max_area_of_island(grid) == 3
