"""Tests for N-Queens kata."""

try:
    from user_kata import solve_n_queens
except ImportError:
    from .reference import solve_n_queens


def test_solve_n_queens_example1():
    result = solve_n_queens(4)
    result = [sorted(board) for board in result]
    result = sorted(result)
    expected = [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    expected = [sorted(board) for board in expected]
    expected = sorted(expected)
    assert result == expected

def test_solve_n_queens_example2():
    assert solve_n_queens(1) == [["Q"]]

def test_solve_n_queens_n2():
    result = solve_n_queens(2)
    assert result == []

def test_solve_n_queens_n3():
    result = solve_n_queens(3)
    assert result == []
