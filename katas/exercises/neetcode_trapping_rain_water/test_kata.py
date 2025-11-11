"""Tests for Trapping Rain Water kata."""

try:
    from user_kata import trap
except ImportError:
    from .reference import trap


def test_trap_example1():
    assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6

def test_trap_example2():
    assert trap([4,2,0,3,2,5]) == 9

def test_trap_no_water():
    assert trap([1,2,3,4,5]) == 0
