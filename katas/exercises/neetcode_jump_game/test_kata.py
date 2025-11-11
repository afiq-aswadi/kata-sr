"""Tests for Jump Game kata."""

try:
    from user_kata import can_jump
except ImportError:
    from .reference import can_jump


def test_can_jump_example1():
    assert can_jump([2,3,1,1,4]) == True

def test_can_jump_example2():
    assert can_jump([3,2,1,0,4]) == False

def test_can_jump_single():
    assert can_jump([0]) == True

def test_can_jump_impossible():
    assert can_jump([0,1]) == False

def test_can_jump_possible():
    assert can_jump([2,0,0]) == True
