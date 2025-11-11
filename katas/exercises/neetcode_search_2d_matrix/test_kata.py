"""Tests for Search a 2D Matrix kata."""

try:
    from user_kata import search_matrix
except ImportError:
    from .reference import search_matrix


def test_search_matrix_found():
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 3) == True

def test_search_matrix_not_found():
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 13) == False
