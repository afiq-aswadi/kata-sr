"""Tests for Diameter of Binary Tree kata."""

try:
    from user_kata import TreeNode, diameter_of_binary_tree
except ImportError:
    from .reference import TreeNode, diameter_of_binary_tree


def test_diameter_example1():

    # Create tree [1,2,3,4,5]
    root = TreeNode(1)
    root.left = TreeNode(2, TreeNode(4), TreeNode(5))
    root.right = TreeNode(3)

    assert diameter_of_binary_tree(root) == 3

def test_diameter_example2():

    # Create tree [1,2]
    root = TreeNode(1, TreeNode(2))

    assert diameter_of_binary_tree(root) == 1

def test_diameter_single():
    root = TreeNode(1)
    assert diameter_of_binary_tree(root) == 0

def test_diameter_skewed():

    # Create left-skewed tree [1,2,null,3]
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)

    assert diameter_of_binary_tree(root) == 2

def test_diameter_balanced():

    # Create balanced tree
    root = TreeNode(1)
    root.left = TreeNode(2, TreeNode(4), TreeNode(5))
    root.right = TreeNode(3, TreeNode(6), TreeNode(7))

    assert diameter_of_binary_tree(root) == 4
