"""Tests for Word Search kata."""

try:
    from user_kata import exist
except ImportError:
    from .reference import exist


def test_exist_example1():
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "ABCCED") == True

def test_exist_example2():
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "SEE") == True

def test_exist_example3():
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "ABCB") == False

def test_exist_single_cell():
    board = [["A"]]
    assert exist(board, "A") == True
    assert exist(board, "B") == False
