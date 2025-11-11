"""Tests for Count Good Nodes in Binary Tree kata."""

try:
    from user_kata import TreeNode, good_nodes
except ImportError:
    from .reference import TreeNode, good_nodes


def test_good_nodes_example1():

    # Create tree [3,1,4,3,null,1,5]
    root = TreeNode(3)
    root.left = TreeNode(1, TreeNode(3))
    root.right = TreeNode(4, TreeNode(1), TreeNode(5))

    assert good_nodes(root) == 4

def test_good_nodes_example2():

    # Create tree [3,3,null,4,2]
    root = TreeNode(3)
    root.left = TreeNode(3, TreeNode(4), TreeNode(2))

    assert good_nodes(root) == 3

def test_good_nodes_example3():

    root = TreeNode(1)

    assert good_nodes(root) == 1

def test_good_nodes_all_increasing():

    # All nodes are good
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)

    assert good_nodes(root) == 3

def test_good_nodes_all_same():

    # All nodes have same value
    root = TreeNode(5)
    root.left = TreeNode(5, TreeNode(5))
    root.right = TreeNode(5)

    assert good_nodes(root) == 4
