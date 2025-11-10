"""Tests for Find the Duplicate Number kata."""

def test_find_duplicate_basic():
    from template import find_duplicate
    assert find_duplicate([1, 3, 4, 2, 2]) == 2

def test_find_duplicate_multiple_occurrences():
    from template import find_duplicate
    assert find_duplicate([3, 1, 3, 4, 2]) == 3

def test_find_duplicate_all_same():
    from template import find_duplicate
    assert find_duplicate([3, 3, 3, 3, 3]) == 3

def test_find_duplicate_at_beginning():
    from template import find_duplicate
    assert find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1, 4]) == 9

def test_find_duplicate_small():
    from template import find_duplicate
    assert find_duplicate([1, 1]) == 1
