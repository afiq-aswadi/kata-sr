"""Tests for Number of Islands kata."""

try:
    from user_kata import num_islands
except ImportError:
    from .reference import num_islands


def test_num_islands_example1():
    grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    assert num_islands(grid) == 1

def test_num_islands_example2():
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    assert num_islands(grid) == 3

def test_num_islands_single():
    grid = [["1"]]
    assert num_islands(grid) == 1

def test_num_islands_no_land():
    grid = [["0","0"],["0","0"]]
    assert num_islands(grid) == 0
