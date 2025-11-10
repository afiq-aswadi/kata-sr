"""Tests for House Robber kata."""

def test_rob_example1():
    from template import rob
    assert rob([1,2,3,1]) == 4

def test_rob_example2():
    from template import rob
    assert rob([2,7,9,3,1]) == 12

def test_rob_single_house():
    from template import rob
    assert rob([5]) == 5

def test_rob_two_houses():
    from template import rob
    assert rob([1,2]) == 2
    assert rob([2,1]) == 2
