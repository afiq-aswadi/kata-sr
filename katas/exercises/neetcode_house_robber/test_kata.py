"""Tests for House Robber kata."""

try:
    from user_kata import rob
except ImportError:
    from .reference import rob


def test_rob_example1():
    assert rob([1,2,3,1]) == 4

def test_rob_example2():
    assert rob([2,7,9,3,1]) == 12

def test_rob_single_house():
    assert rob([5]) == 5

def test_rob_two_houses():
    assert rob([1,2]) == 2
    assert rob([2,1]) == 2
