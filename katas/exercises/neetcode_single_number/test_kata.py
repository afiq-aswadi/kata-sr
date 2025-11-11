"""Tests for Single Number kata."""

try:
    from user_kata import single_number
except ImportError:
    from .reference import single_number


def test_single_number_example1():
    assert single_number([2,2,1]) == 1

def test_single_number_example2():
    assert single_number([4,1,2,1,2]) == 4

def test_single_number_example3():
    assert single_number([1]) == 1

def test_single_number_negative():
    assert single_number([-1,-1,2]) == 2

def test_single_number_multiple_pairs():
    assert single_number([1,1,2,2,3,3,4]) == 4
