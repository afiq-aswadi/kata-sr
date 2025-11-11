"""Tests for Search in Rotated Sorted Array kata."""

try:
    from user_kata import search
except ImportError:
    from .reference import search


def test_search_rotated_found():
    assert search([4,5,6,7,0,1,2], 0) == 4

def test_search_rotated_not_found():
    assert search([4,5,6,7,0,1,2], 3) == -1

def test_search_rotated_single():
    assert search([1], 1) == 0
