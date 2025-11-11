"""Tests for Binary Search kata."""

try:
    from user_kata import search
except ImportError:
    from .reference import search


def test_search_found():
    assert search([-1,0,3,5,9,12], 9) == 4

def test_search_not_found():
    assert search([-1,0,3,5,9,12], 2) == -1

def test_search_single():
    assert search([5], 5) == 0
