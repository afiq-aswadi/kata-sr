"""Tests for N-Queens kata."""

def test_solve_n_queens_example1():
    from template import solve_n_queens
    result = solve_n_queens(4)
    result = [sorted(board) for board in result]
    result = sorted(result)
    expected = [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    expected = [sorted(board) for board in expected]
    expected = sorted(expected)
    assert result == expected

def test_solve_n_queens_example2():
    from template import solve_n_queens
    assert solve_n_queens(1) == [["Q"]]

def test_solve_n_queens_n2():
    from template import solve_n_queens
    result = solve_n_queens(2)
    assert result == []

def test_solve_n_queens_n3():
    from template import solve_n_queens
    result = solve_n_queens(3)
    assert result == []
