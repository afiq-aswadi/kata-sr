"""Tests for Number of Islands kata."""

def test_num_islands_example1():
    from template import num_islands
    grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
    ]
    assert num_islands(grid) == 1

def test_num_islands_example2():
    from template import num_islands
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    assert num_islands(grid) == 3

def test_num_islands_single():
    from template import num_islands
    grid = [["1"]]
    assert num_islands(grid) == 1

def test_num_islands_no_land():
    from template import num_islands
    grid = [["0","0"],["0","0"]]
    assert num_islands(grid) == 0
