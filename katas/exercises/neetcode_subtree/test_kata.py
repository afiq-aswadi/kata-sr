"""Tests for Subtree of Another Tree kata."""

try:
    from user_kata import TreeNode, is_subtree
except ImportError:
    from .reference import TreeNode, is_subtree


def test_subtree_example1():

    # Create root tree [3,4,5,1,2]
    root = TreeNode(3)
    root.left = TreeNode(4, TreeNode(1), TreeNode(2))
    root.right = TreeNode(5)

    # Create subRoot tree [4,1,2]
    sub_root = TreeNode(4, TreeNode(1), TreeNode(2))

    assert is_subtree(root, sub_root) == True

def test_subtree_example2():

    # Create root tree [3,4,5,1,2,null,null,null,null,0]
    root = TreeNode(3)
    root.left = TreeNode(4, TreeNode(1), TreeNode(2, TreeNode(0)))
    root.right = TreeNode(5)

    # Create subRoot tree [4,1,2]
    sub_root = TreeNode(4, TreeNode(1), TreeNode(2))

    assert is_subtree(root, sub_root) == False

def test_subtree_identical():

    # Create identical trees [1,2,3]
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(1, TreeNode(2), TreeNode(3))

    assert is_subtree(root, sub_root) == True

def test_subtree_single_node():

    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(2)

    assert is_subtree(root, sub_root) == True

def test_subtree_not_found():

    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(4)

    assert is_subtree(root, sub_root) == False
