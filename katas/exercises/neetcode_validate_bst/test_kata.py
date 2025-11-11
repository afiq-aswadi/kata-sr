"""Tests for Validate Binary Search Tree kata."""

try:
    from user_kata import TreeNode, is_valid_bst
except ImportError:
    from .reference import TreeNode, is_valid_bst


def test_validate_bst_example1():

    # Create BST [2,1,3]
    root = TreeNode(2, TreeNode(1), TreeNode(3))

    assert is_valid_bst(root) == True

def test_validate_bst_example2():

    # Create invalid BST [5,1,4,null,null,3,6]
    root = TreeNode(5)
    root.left = TreeNode(1)
    root.right = TreeNode(4, TreeNode(3), TreeNode(6))

    assert is_valid_bst(root) == False

def test_validate_bst_single():
    root = TreeNode(1)
    assert is_valid_bst(root) == True

def test_validate_bst_valid():

    # Create valid BST [10,5,15,null,null,12,20]
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15, TreeNode(12), TreeNode(20))

    assert is_valid_bst(root) == True

def test_validate_bst_equal_values():

    # BST with equal values is invalid
    root = TreeNode(5, TreeNode(5))

    assert is_valid_bst(root) == False
