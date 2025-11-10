"""Tests for Binary Tree Right Side View kata."""

def test_right_side_view_example1():
    from template import TreeNode, right_side_view

    # Create tree [1,2,3,null,5,null,4]
    root = TreeNode(1)
    root.left = TreeNode(2, None, TreeNode(5))
    root.right = TreeNode(3, None, TreeNode(4))

    assert right_side_view(root) == [1, 3, 4]

def test_right_side_view_example2():
    from template import TreeNode, right_side_view

    # Create tree [1,null,3]
    root = TreeNode(1, None, TreeNode(3))

    assert right_side_view(root) == [1, 3]

def test_right_side_view_empty():
    from template import right_side_view
    assert right_side_view(None) == []

def test_right_side_view_single():
    from template import TreeNode, right_side_view
    root = TreeNode(1)
    assert right_side_view(root) == [1]

def test_right_side_view_left_skewed():
    from template import TreeNode, right_side_view

    # Create left-skewed tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)

    assert right_side_view(root) == [1, 2, 3]
