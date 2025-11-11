"""Tests for Binary Tree Level Order Traversal kata."""

try:
    from user_kata import TreeNode, level_order
    from user_kata import level_order
except ImportError:
    from .reference import TreeNode, level_order
    from .reference import level_order


def test_level_order_example1():

    # Create tree [3,9,20,null,null,15,7]
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    assert level_order(root) == [[3], [9, 20], [15, 7]]

def test_level_order_example2():

    root = TreeNode(1)

    assert level_order(root) == [[1]]

def test_level_order_empty():
    assert level_order(None) == []

def test_level_order_skewed():

    # Create left-skewed tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)

    assert level_order(root) == [[1], [2], [3]]

def test_level_order_complete():

    # Create complete tree [1,2,3,4,5,6,7]
    root = TreeNode(1)
    root.left = TreeNode(2, TreeNode(4), TreeNode(5))
    root.right = TreeNode(3, TreeNode(6), TreeNode(7))

    assert level_order(root) == [[1], [2, 3], [4, 5, 6, 7]]
