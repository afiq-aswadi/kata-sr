"""Tests for Surrounded Regions kata."""

def test_solve_example1():
    from template import solve
    board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    solve(board)
    expected = [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    assert board == expected

def test_solve_example2():
    from template import solve
    board = [["X"]]
    solve(board)
    expected = [["X"]]
    assert board == expected

def test_solve_all_border():
    from template import solve
    board = [["O","O","O"],["O","X","O"],["O","O","O"]]
    solve(board)
    expected = [["O","O","O"],["O","X","O"],["O","O","O"]]
    assert board == expected

def test_solve_simple():
    from template import solve
    board = [["X","X","X"],["X","O","X"],["X","X","X"]]
    solve(board)
    expected = [["X","X","X"],["X","X","X"],["X","X","X"]]
    assert board == expected
