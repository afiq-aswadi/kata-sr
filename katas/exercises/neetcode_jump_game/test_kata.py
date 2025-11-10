"""Tests for Jump Game kata."""

def test_can_jump_example1():
    from template import can_jump
    assert can_jump([2,3,1,1,4]) == True

def test_can_jump_example2():
    from template import can_jump
    assert can_jump([3,2,1,0,4]) == False

def test_can_jump_single():
    from template import can_jump
    assert can_jump([0]) == True

def test_can_jump_impossible():
    from template import can_jump
    assert can_jump([0,1]) == False

def test_can_jump_possible():
    from template import can_jump
    assert can_jump([2,0,0]) == True
