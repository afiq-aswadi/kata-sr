"""Tests for Minimum Interval to Include Each Query kata."""

try:
    from user_kata import min_interval
except ImportError:
    from .reference import min_interval


def test_min_interval_example1():
    assert min_interval([[1,4],[2,4],[3,6],[4,4]], [2,3,4,5]) == [3,3,1,4]

def test_min_interval_example2():
    assert min_interval([[2,3],[2,5],[1,8],[20,25]], [2,19,5,22]) == [2,-1,4,6]

def test_min_interval_single():
    assert min_interval([[1,5]], [3]) == [5]

def test_min_interval_no_match():
    assert min_interval([[1,2],[3,4]], [5]) == [-1]

def test_min_interval_multiple_queries():
    assert min_interval([[1,10]], [1,5,10]) == [10,10,10]
