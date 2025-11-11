"""Tests for Non-overlapping Intervals kata."""

try:
    from user_kata import erase_overlap_intervals
except ImportError:
    from .reference import erase_overlap_intervals


def test_erase_overlap_intervals_example1():
    assert erase_overlap_intervals([[1,2],[2,3],[3,4],[1,3]]) == 1

def test_erase_overlap_intervals_example2():
    assert erase_overlap_intervals([[1,2],[1,2],[1,2]]) == 2

def test_erase_overlap_intervals_example3():
    assert erase_overlap_intervals([[1,2],[2,3]]) == 0

def test_erase_overlap_intervals_single():
    assert erase_overlap_intervals([[1,2]]) == 0

def test_erase_overlap_intervals_nested():
    assert erase_overlap_intervals([[1,100],[11,22],[1,11],[2,12]]) == 2
