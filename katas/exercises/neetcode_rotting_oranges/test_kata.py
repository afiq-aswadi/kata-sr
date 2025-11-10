"""Tests for Rotting Oranges kata."""

def test_oranges_rotting_example1():
    from template import oranges_rotting
    grid = [[2,1,1],[1,1,0],[0,1,1]]
    assert oranges_rotting(grid) == 4

def test_oranges_rotting_example2():
    from template import oranges_rotting
    grid = [[2,1,1],[0,1,1],[1,0,1]]
    assert oranges_rotting(grid) == -1

def test_oranges_rotting_example3():
    from template import oranges_rotting
    grid = [[0,2]]
    assert oranges_rotting(grid) == 0

def test_oranges_rotting_no_fresh():
    from template import oranges_rotting
    grid = [[2,2],[2,2]]
    assert oranges_rotting(grid) == 0

def test_oranges_rotting_single():
    from template import oranges_rotting
    grid = [[2,1]]
    assert oranges_rotting(grid) == 1
