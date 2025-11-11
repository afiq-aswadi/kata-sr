"""Tests for N-Queens II kata."""

try:
    from user_kata import total_n_queens
except ImportError:
    from .reference import total_n_queens


def test_total_n_queens_example1():
    assert total_n_queens(4) == 2

def test_total_n_queens_example2():
    assert total_n_queens(1) == 1

def test_total_n_queens_n2():
    assert total_n_queens(2) == 0

def test_total_n_queens_n3():
    assert total_n_queens(3) == 0

def test_total_n_queens_n5():
    assert total_n_queens(5) == 10
