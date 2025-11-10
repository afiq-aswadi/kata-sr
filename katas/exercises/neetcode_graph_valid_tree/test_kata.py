"""Tests for Graph Valid Tree kata."""

def test_valid_tree_example1():
    from template import valid_tree
    assert valid_tree(5, [[0,1],[0,2],[0,3],[1,4]]) == True

def test_valid_tree_example2():
    from template import valid_tree
    assert valid_tree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]) == False

def test_valid_tree_single_node():
    from template import valid_tree
    assert valid_tree(1, []) == True

def test_valid_tree_two_nodes():
    from template import valid_tree
    assert valid_tree(2, [[0,1]]) == True

def test_valid_tree_disconnected():
    from template import valid_tree
    assert valid_tree(4, [[0,1],[2,3]]) == False
