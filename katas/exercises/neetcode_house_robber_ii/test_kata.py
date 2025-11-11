"""Tests for House Robber II kata."""

try:
    from user_kata import rob
except ImportError:
    from .reference import rob


def test_rob_example1():
    assert rob([2,3,2]) == 3

def test_rob_example2():
    assert rob([1,2,3,1]) == 4

def test_rob_example3():
    assert rob([1,2,3]) == 3

def test_rob_single_house():
    assert rob([1]) == 1
