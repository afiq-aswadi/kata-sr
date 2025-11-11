"""Tests for Surrounded Regions kata."""

try:
    from user_kata import solve
except ImportError:
    from .reference import solve


def test_solve_example1():
    board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    solve(board)
    expected = [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    assert board == expected

def test_solve_example2():
    board = [["X"]]
    solve(board)
    expected = [["X"]]
    assert board == expected

def test_solve_all_border():
    board = [["O","O","O"],["O","X","O"],["O","O","O"]]
    solve(board)
    expected = [["O","O","O"],["O","X","O"],["O","O","O"]]
    assert board == expected

def test_solve_simple():
    board = [["X","X","X"],["X","O","X"],["X","X","X"]]
    solve(board)
    expected = [["X","X","X"],["X","X","X"],["X","X","X"]]
    assert board == expected
