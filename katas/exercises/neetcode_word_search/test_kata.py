"""Tests for Word Search kata."""

def test_exist_example1():
    from template import exist
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "ABCCED") == True

def test_exist_example2():
    from template import exist
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "SEE") == True

def test_exist_example3():
    from template import exist
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert exist(board, "ABCB") == False

def test_exist_single_cell():
    from template import exist
    board = [["A"]]
    assert exist(board, "A") == True
    assert exist(board, "B") == False
