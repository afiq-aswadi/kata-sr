"""Tests for Longest Consecutive Sequence kata."""

try:
    from user_kata import longest_consecutive
except ImportError:
    from .reference import longest_consecutive


def test_longest_consecutive_example1():
    assert longest_consecutive([100,4,200,1,3,2]) == 4

def test_longest_consecutive_example2():
    assert longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9

def test_longest_consecutive_empty():
    assert longest_consecutive([]) == 0

def test_longest_consecutive_single():
    assert longest_consecutive([1]) == 1
