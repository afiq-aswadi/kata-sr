"""Tests for Balanced Binary Tree kata."""

def test_balanced_example1():
    from template import TreeNode, is_balanced

    # Create tree [3,9,20,null,null,15,7]
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    assert is_balanced(root) == True

def test_balanced_example2():
    from template import TreeNode, is_balanced

    # Create tree [1,2,2,3,3,null,null,4,4]
    root = TreeNode(1)
    root.left = TreeNode(2, TreeNode(3, TreeNode(4), TreeNode(4)), TreeNode(3))
    root.right = TreeNode(2)

    assert is_balanced(root) == False

def test_balanced_empty():
    from template import is_balanced
    assert is_balanced(None) == True

def test_balanced_single():
    from template import TreeNode, is_balanced
    root = TreeNode(1)
    assert is_balanced(root) == True

def test_balanced_skewed():
    from template import TreeNode, is_balanced

    # Create left-skewed tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)

    assert is_balanced(root) == False
