"""Tests for Jump Game II kata."""

def test_jump_example1():
    from template import jump
    assert jump([2,3,1,1,4]) == 2

def test_jump_example2():
    from template import jump
    assert jump([2,3,0,1,4]) == 2

def test_jump_single():
    from template import jump
    assert jump([0]) == 0

def test_jump_two_elements():
    from template import jump
    assert jump([1,2]) == 1

def test_jump_long_distance():
    from template import jump
    assert jump([5,9,3,2,1,0,2,3,3,1,0,0]) == 3
