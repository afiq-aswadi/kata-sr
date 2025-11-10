"""Tests for Same Tree kata."""

def test_same_tree_example1():
    from template import TreeNode, is_same_tree

    # Create trees [1,2,3] and [1,2,3]
    p = TreeNode(1, TreeNode(2), TreeNode(3))
    q = TreeNode(1, TreeNode(2), TreeNode(3))

    assert is_same_tree(p, q) == True

def test_same_tree_example2():
    from template import TreeNode, is_same_tree

    # Create trees [1,2] and [1,null,2]
    p = TreeNode(1, TreeNode(2))
    q = TreeNode(1, None, TreeNode(2))

    assert is_same_tree(p, q) == False

def test_same_tree_example3():
    from template import TreeNode, is_same_tree

    # Create trees [1,2,1] and [1,1,2]
    p = TreeNode(1, TreeNode(2), TreeNode(1))
    q = TreeNode(1, TreeNode(1), TreeNode(2))

    assert is_same_tree(p, q) == False

def test_same_tree_both_empty():
    from template import is_same_tree
    assert is_same_tree(None, None) == True

def test_same_tree_one_empty():
    from template import TreeNode, is_same_tree
    p = TreeNode(1)
    assert is_same_tree(p, None) == False
    assert is_same_tree(None, p) == False
