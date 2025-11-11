"""Tests for Plus One kata."""

try:
    from user_kata import plus_one
except ImportError:
    from .reference import plus_one


def test_plus_one_example1():
    assert plus_one([1,2,3]) == [1,2,4]

def test_plus_one_example2():
    assert plus_one([4,3,2,1]) == [4,3,2,2]

def test_plus_one_example3():
    assert plus_one([9]) == [1,0]

def test_plus_one_multiple_nines():
    assert plus_one([9,9,9]) == [1,0,0,0]

def test_plus_one_with_carry():
    assert plus_one([1,9,9]) == [2,0,0]
