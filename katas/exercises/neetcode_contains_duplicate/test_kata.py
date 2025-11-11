"""Tests for Contains Duplicate kata."""

try:
    from user_kata import contains_duplicate
except ImportError:
    from .reference import contains_duplicate


def test_contains_duplicate_example1():
    assert contains_duplicate([1,2,3,1]) == True

def test_contains_duplicate_example2():
    assert contains_duplicate([1,2,3,4]) == False

def test_contains_duplicate_empty():
    assert contains_duplicate([1]) == False
