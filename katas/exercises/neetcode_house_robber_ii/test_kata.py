"""Tests for House Robber II kata."""

def test_rob_example1():
    from template import rob
    assert rob([2,3,2]) == 3

def test_rob_example2():
    from template import rob
    assert rob([1,2,3,1]) == 4

def test_rob_example3():
    from template import rob
    assert rob([1,2,3]) == 3

def test_rob_single_house():
    from template import rob
    assert rob([1]) == 1
