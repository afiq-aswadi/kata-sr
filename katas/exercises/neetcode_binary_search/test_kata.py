"""Tests for Binary Search kata."""

def test_search_found():
    from template import search
    assert search([-1,0,3,5,9,12], 9) == 4

def test_search_not_found():
    from template import search
    assert search([-1,0,3,5,9,12], 2) == -1

def test_search_single():
    from template import search
    assert search([5], 5) == 0
