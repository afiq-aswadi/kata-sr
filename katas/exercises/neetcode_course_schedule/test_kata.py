"""Tests for Course Schedule kata."""

def test_can_finish_example1():
    from template import can_finish
    assert can_finish(2, [[1,0]]) == True

def test_can_finish_example2():
    from template import can_finish
    assert can_finish(2, [[1,0],[0,1]]) == False

def test_can_finish_simple():
    from template import can_finish
    assert can_finish(3, [[0,1],[0,2],[1,2]]) == True

def test_can_finish_no_prereq():
    from template import can_finish
    assert can_finish(5, []) == True

def test_can_finish_complex():
    from template import can_finish
    assert can_finish(4, [[1,0],[2,0],[3,1],[3,2]]) == True
