"""Tests for Search a 2D Matrix kata."""

def test_search_matrix_found():
    from template import search_matrix
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 3) == True

def test_search_matrix_not_found():
    from template import search_matrix
    matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
    assert search_matrix(matrix, 13) == False
