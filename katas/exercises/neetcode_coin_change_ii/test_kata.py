"""Tests for Coin Change II kata."""

try:
    from user_kata import change
except ImportError:
    from .reference import change


def test_change_example1():
    assert change(5, [1,2,5]) == 4

def test_change_example2():
    assert change(3, [2]) == 0

def test_change_example3():
    assert change(10, [10]) == 1

def test_change_zero():
    assert change(0, [1,2,5]) == 1

def test_change_small():
    assert change(2, [1,2]) == 2
