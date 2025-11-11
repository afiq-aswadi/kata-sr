"""Tests for Missing Number kata."""

try:
    from user_kata import missing_number
except ImportError:
    from .reference import missing_number


def test_missing_number_example1():
    assert missing_number([3,0,1]) == 2

def test_missing_number_example2():
    assert missing_number([0,1]) == 2

def test_missing_number_example3():
    assert missing_number([9,6,4,2,3,5,7,0,1]) == 8

def test_missing_number_zero():
    assert missing_number([1]) == 0

def test_missing_number_last():
    assert missing_number([0,1,2,3]) == 4
