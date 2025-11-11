"""Tests for Set Matrix Zeroes kata."""

try:
    from user_kata import set_zeroes
except ImportError:
    from .reference import set_zeroes


def test_set_zeroes_example1():
    matrix = [[1,1,1],[1,0,1],[1,1,1]]
    set_zeroes(matrix)
    assert matrix == [[1,0,1],[0,0,0],[1,0,1]]

def test_set_zeroes_example2():
    matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    set_zeroes(matrix)
    assert matrix == [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

def test_set_zeroes_single():
    matrix = [[0]]
    set_zeroes(matrix)
    assert matrix == [[0]]

def test_set_zeroes_no_zeros():
    matrix = [[1,2],[3,4]]
    set_zeroes(matrix)
    assert matrix == [[1,2],[3,4]]
