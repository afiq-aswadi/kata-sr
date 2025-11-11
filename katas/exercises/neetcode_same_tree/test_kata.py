"""Tests for Same Tree kata."""

try:
    from user_kata import TreeNode, is_same_tree
    from user_kata import is_same_tree
except ImportError:
    from .reference import TreeNode, is_same_tree
    from .reference import is_same_tree


def test_same_tree_example1():

    # Create trees [1,2,3] and [1,2,3]
    p = TreeNode(1, TreeNode(2), TreeNode(3))
    q = TreeNode(1, TreeNode(2), TreeNode(3))

    assert is_same_tree(p, q) == True

def test_same_tree_example2():

    # Create trees [1,2] and [1,null,2]
    p = TreeNode(1, TreeNode(2))
    q = TreeNode(1, None, TreeNode(2))

    assert is_same_tree(p, q) == False

def test_same_tree_example3():

    # Create trees [1,2,1] and [1,1,2]
    p = TreeNode(1, TreeNode(2), TreeNode(1))
    q = TreeNode(1, TreeNode(1), TreeNode(2))

    assert is_same_tree(p, q) == False

def test_same_tree_both_empty():
    assert is_same_tree(None, None) == True

def test_same_tree_one_empty():
    p = TreeNode(1)
    assert is_same_tree(p, None) == False
    assert is_same_tree(None, p) == False
