"""Tests for N-Queens II kata."""

def test_total_n_queens_example1():
    from template import total_n_queens
    assert total_n_queens(4) == 2

def test_total_n_queens_example2():
    from template import total_n_queens
    assert total_n_queens(1) == 1

def test_total_n_queens_n2():
    from template import total_n_queens
    assert total_n_queens(2) == 0

def test_total_n_queens_n3():
    from template import total_n_queens
    assert total_n_queens(3) == 0

def test_total_n_queens_n5():
    from template import total_n_queens
    assert total_n_queens(5) == 10
