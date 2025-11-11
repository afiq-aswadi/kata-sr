"""Tests for Redundant Connection kata."""

try:
    from user_kata import find_redundant_connection
except ImportError:
    from .reference import find_redundant_connection


def test_find_redundant_connection_example1():
    assert find_redundant_connection([[1,2],[1,3],[2,3]]) == [2,3]

def test_find_redundant_connection_example2():
    assert find_redundant_connection([[1,2],[2,3],[3,4],[1,4],[1,5]]) == [1,4]

def test_find_redundant_connection_simple():
    assert find_redundant_connection([[1,2],[2,3],[1,3]]) == [1,3]

def test_find_redundant_connection_linear():
    assert find_redundant_connection([[1,2],[2,3],[3,4],[4,5],[2,5]]) == [2,5]
