"""Tests for Subtree of Another Tree kata."""

def test_subtree_example1():
    from template import TreeNode, is_subtree

    # Create root tree [3,4,5,1,2]
    root = TreeNode(3)
    root.left = TreeNode(4, TreeNode(1), TreeNode(2))
    root.right = TreeNode(5)

    # Create subRoot tree [4,1,2]
    sub_root = TreeNode(4, TreeNode(1), TreeNode(2))

    assert is_subtree(root, sub_root) == True

def test_subtree_example2():
    from template import TreeNode, is_subtree

    # Create root tree [3,4,5,1,2,null,null,null,null,0]
    root = TreeNode(3)
    root.left = TreeNode(4, TreeNode(1), TreeNode(2, TreeNode(0)))
    root.right = TreeNode(5)

    # Create subRoot tree [4,1,2]
    sub_root = TreeNode(4, TreeNode(1), TreeNode(2))

    assert is_subtree(root, sub_root) == False

def test_subtree_identical():
    from template import TreeNode, is_subtree

    # Create identical trees [1,2,3]
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(1, TreeNode(2), TreeNode(3))

    assert is_subtree(root, sub_root) == True

def test_subtree_single_node():
    from template import TreeNode, is_subtree

    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(2)

    assert is_subtree(root, sub_root) == True

def test_subtree_not_found():
    from template import TreeNode, is_subtree

    root = TreeNode(1, TreeNode(2), TreeNode(3))
    sub_root = TreeNode(4)

    assert is_subtree(root, sub_root) == False
