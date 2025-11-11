"""Tests for Find the Duplicate Number kata."""

try:
    from user_kata import find_duplicate
except ImportError:
    from .reference import find_duplicate


def test_find_duplicate_basic():
    assert find_duplicate([1, 3, 4, 2, 2]) == 2

def test_find_duplicate_multiple_occurrences():
    assert find_duplicate([3, 1, 3, 4, 2]) == 3

def test_find_duplicate_all_same():
    assert find_duplicate([3, 3, 3, 3, 3]) == 3

def test_find_duplicate_at_beginning():
    assert find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1, 4]) == 9

def test_find_duplicate_small():
    assert find_duplicate([1, 1]) == 1
