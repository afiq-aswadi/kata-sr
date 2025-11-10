"""Tests for Contains Duplicate kata."""

def test_contains_duplicate_example1():
    from template import contains_duplicate
    assert contains_duplicate([1,2,3,1]) == True

def test_contains_duplicate_example2():
    from template import contains_duplicate
    assert contains_duplicate([1,2,3,4]) == False

def test_contains_duplicate_empty():
    from template import contains_duplicate
    assert contains_duplicate([1]) == False
