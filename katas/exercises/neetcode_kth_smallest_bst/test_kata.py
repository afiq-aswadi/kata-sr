"""Tests for Kth Smallest Element in a BST kata."""

def test_kth_smallest_example1():
    from template import TreeNode, kth_smallest

    # Create BST [3,1,4,null,2]
    root = TreeNode(3)
    root.left = TreeNode(1, None, TreeNode(2))
    root.right = TreeNode(4)

    assert kth_smallest(root, 1) == 1

def test_kth_smallest_example2():
    from template import TreeNode, kth_smallest

    # Create BST [5,3,6,2,4,null,null,1]
    root = TreeNode(5)
    root.left = TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4))
    root.right = TreeNode(6)

    assert kth_smallest(root, 3) == 3

def test_kth_smallest_first():
    from template import TreeNode, kth_smallest

    # Create BST [2,1,3]
    root = TreeNode(2, TreeNode(1), TreeNode(3))

    assert kth_smallest(root, 1) == 1

def test_kth_smallest_last():
    from template import TreeNode, kth_smallest

    # Create BST [2,1,3]
    root = TreeNode(2, TreeNode(1), TreeNode(3))

    assert kth_smallest(root, 3) == 3

def test_kth_smallest_middle():
    from template import TreeNode, kth_smallest

    # Create BST [4,2,6,1,3,5,7]
    root = TreeNode(4)
    root.left = TreeNode(2, TreeNode(1), TreeNode(3))
    root.right = TreeNode(6, TreeNode(5), TreeNode(7))

    assert kth_smallest(root, 4) == 4
