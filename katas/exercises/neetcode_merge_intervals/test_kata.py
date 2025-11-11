"""Tests for Merge Intervals kata."""

try:
    from user_kata import merge
except ImportError:
    from .reference import merge


def test_merge_example1():
    assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]

def test_merge_example2():
    assert merge([[1,4],[4,5]]) == [[1,5]]

def test_merge_single():
    assert merge([[1,4]]) == [[1,4]]

def test_merge_no_overlap():
    assert merge([[1,2],[3,4]]) == [[1,2],[3,4]]

def test_merge_all_overlap():
    assert merge([[1,10],[2,6],[8,9]]) == [[1,10]]
