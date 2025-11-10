"""Tests for Merge Intervals kata."""

def test_merge_example1():
    from template import merge
    assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]

def test_merge_example2():
    from template import merge
    assert merge([[1,4],[4,5]]) == [[1,5]]

def test_merge_single():
    from template import merge
    assert merge([[1,4]]) == [[1,4]]

def test_merge_no_overlap():
    from template import merge
    assert merge([[1,2],[3,4]]) == [[1,2],[3,4]]

def test_merge_all_overlap():
    from template import merge
    assert merge([[1,10],[2,6],[8,9]]) == [[1,10]]
