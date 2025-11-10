"""Tests for Lowest Common Ancestor of a Binary Search Tree kata."""

def test_lca_bst_example1():
    from template import TreeNode, lowest_common_ancestor

    # Create BST [6,2,8,0,4,7,9,null,null,3,5]
    root = TreeNode(6)
    root.left = TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5)))
    root.right = TreeNode(8, TreeNode(7), TreeNode(9))

    p = root.left  # 2
    q = root.right  # 8

    assert lowest_common_ancestor(root, p, q) == root

def test_lca_bst_example2():
    from template import TreeNode, lowest_common_ancestor

    # Create BST [6,2,8,0,4,7,9,null,null,3,5]
    root = TreeNode(6)
    root.left = TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5)))
    root.right = TreeNode(8, TreeNode(7), TreeNode(9))

    p = root.left  # 2
    q = root.left.right  # 4

    assert lowest_common_ancestor(root, p, q) == p

def test_lca_bst_same_subtree():
    from template import TreeNode, lowest_common_ancestor

    # Create BST [6,2,8,0,4,7,9,null,null,3,5]
    root = TreeNode(6)
    root.left = TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5)))
    root.right = TreeNode(8, TreeNode(7), TreeNode(9))

    p = root.left.right.left  # 3
    q = root.left.right.right  # 5

    assert lowest_common_ancestor(root, p, q) == root.left.right

def test_lca_bst_simple():
    from template import TreeNode, lowest_common_ancestor

    # Create BST [2,1,3]
    root = TreeNode(2, TreeNode(1), TreeNode(3))

    p = root.left  # 1
    q = root.right  # 3

    assert lowest_common_ancestor(root, p, q) == root
