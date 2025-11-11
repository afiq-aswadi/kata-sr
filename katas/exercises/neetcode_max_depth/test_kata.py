"""Tests for Maximum Depth of Binary Tree kata."""

try:
    from user_kata import TreeNode, max_depth
    from user_kata import max_depth
except ImportError:
    from .reference import TreeNode, max_depth
    from .reference import max_depth


def test_max_depth_example1():

    # Create tree [3,9,20,null,null,15,7]
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    assert max_depth(root) == 3

def test_max_depth_example2():

    # Create tree [1,null,2]
    root = TreeNode(1)
    root.right = TreeNode(2)

    assert max_depth(root) == 2

def test_max_depth_empty():
    assert max_depth(None) == 0

def test_max_depth_single():
    root = TreeNode(1)
    assert max_depth(root) == 1

def test_max_depth_skewed():

    # Create left-skewed tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.left.left = TreeNode(4)

    assert max_depth(root) == 4
