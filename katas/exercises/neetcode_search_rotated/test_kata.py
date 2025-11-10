"""Tests for Search in Rotated Sorted Array kata."""

def test_search_rotated_found():
    from template import search
    assert search([4,5,6,7,0,1,2], 0) == 4

def test_search_rotated_not_found():
    from template import search
    assert search([4,5,6,7,0,1,2], 3) == -1

def test_search_rotated_single():
    from template import search
    assert search([1], 1) == 0
