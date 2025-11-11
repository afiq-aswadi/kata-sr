"""Tests for Graph Valid Tree kata."""

try:
    from user_kata import valid_tree
except ImportError:
    from .reference import valid_tree


def test_valid_tree_example1():
    assert valid_tree(5, [[0,1],[0,2],[0,3],[1,4]]) == True

def test_valid_tree_example2():
    assert valid_tree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]) == False

def test_valid_tree_single_node():
    assert valid_tree(1, []) == True

def test_valid_tree_two_nodes():
    assert valid_tree(2, [[0,1]]) == True

def test_valid_tree_disconnected():
    assert valid_tree(4, [[0,1],[2,3]]) == False
