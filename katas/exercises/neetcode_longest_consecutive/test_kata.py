"""Tests for Longest Consecutive Sequence kata."""

def test_longest_consecutive_example1():
    from template import longest_consecutive
    assert longest_consecutive([100,4,200,1,3,2]) == 4

def test_longest_consecutive_example2():
    from template import longest_consecutive
    assert longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9

def test_longest_consecutive_empty():
    from template import longest_consecutive
    assert longest_consecutive([]) == 0

def test_longest_consecutive_single():
    from template import longest_consecutive
    assert longest_consecutive([1]) == 1
