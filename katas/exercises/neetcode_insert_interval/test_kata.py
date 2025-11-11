"""Tests for Insert Interval kata."""

try:
    from user_kata import insert
except ImportError:
    from .reference import insert


def test_insert_example1():
    assert insert([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]

def test_insert_example2():
    assert insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]) == [[1,2],[3,10],[12,16]]

def test_insert_empty():
    assert insert([], [5,7]) == [[5,7]]

def test_insert_no_overlap():
    assert insert([[1,5]], [6,8]) == [[1,5],[6,8]]

def test_insert_complete_overlap():
    assert insert([[3,5],[12,15]], [6,10]) == [[3,5],[6,10],[12,15]]
