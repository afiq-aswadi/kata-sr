"""Tests for Rotate Image kata."""

def test_rotate_example1():
    from template import rotate
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    rotate(matrix)
    assert matrix == [[7,4,1],[8,5,2],[9,6,3]]

def test_rotate_example2():
    from template import rotate
    matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    rotate(matrix)
    assert matrix == [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

def test_rotate_single():
    from template import rotate
    matrix = [[1]]
    rotate(matrix)
    assert matrix == [[1]]

def test_rotate_2x2():
    from template import rotate
    matrix = [[1,2],[3,4]]
    rotate(matrix)
    assert matrix == [[3,1],[4,2]]
