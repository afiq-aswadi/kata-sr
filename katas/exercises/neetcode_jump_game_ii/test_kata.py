"""Tests for Jump Game II kata."""

try:
    from user_kata import jump
except ImportError:
    from .reference import jump


def test_jump_example1():
    assert jump([2,3,1,1,4]) == 2

def test_jump_example2():
    assert jump([2,3,0,1,4]) == 2

def test_jump_single():
    assert jump([0]) == 0

def test_jump_two_elements():
    assert jump([1,2]) == 1

def test_jump_long_distance():
    assert jump([5,9,3,2,1,0,2,3,3,1,0,0]) == 3
