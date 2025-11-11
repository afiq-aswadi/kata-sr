"""Tests for Course Schedule kata."""

try:
    from user_kata import can_finish
except ImportError:
    from .reference import can_finish


def test_can_finish_example1():
    assert can_finish(2, [[1,0]]) == True

def test_can_finish_example2():
    assert can_finish(2, [[1,0],[0,1]]) == False

def test_can_finish_simple():
    assert can_finish(3, [[0,1],[0,2],[1,2]]) == True

def test_can_finish_no_prereq():
    assert can_finish(5, []) == True

def test_can_finish_complex():
    assert can_finish(4, [[1,0],[2,0],[3,1],[3,2]]) == True
