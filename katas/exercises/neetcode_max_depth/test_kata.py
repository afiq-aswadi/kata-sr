"""Tests for Maximum Depth of Binary Tree kata."""

def test_max_depth_example1():
    from template import TreeNode, max_depth

    # Create tree [3,9,20,null,null,15,7]
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    assert max_depth(root) == 3

def test_max_depth_example2():
    from template import TreeNode, max_depth

    # Create tree [1,null,2]
    root = TreeNode(1)
    root.right = TreeNode(2)

    assert max_depth(root) == 2

def test_max_depth_empty():
    from template import max_depth
    assert max_depth(None) == 0

def test_max_depth_single():
    from template import TreeNode, max_depth
    root = TreeNode(1)
    assert max_depth(root) == 1

def test_max_depth_skewed():
    from template import TreeNode, max_depth

    # Create left-skewed tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.left.left = TreeNode(4)

    assert max_depth(root) == 4
