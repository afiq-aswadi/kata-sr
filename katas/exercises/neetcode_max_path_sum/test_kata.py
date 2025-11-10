"""Tests for Binary Tree Maximum Path Sum kata."""

def test_max_path_sum_example1():
    from template import TreeNode, max_path_sum

    # Create tree [1,2,3]
    root = TreeNode(1, TreeNode(2), TreeNode(3))

    assert max_path_sum(root) == 6

def test_max_path_sum_example2():
    from template import TreeNode, max_path_sum

    # Create tree [-10,9,20,null,null,15,7]
    root = TreeNode(-10)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    assert max_path_sum(root) == 42

def test_max_path_sum_single():
    from template import TreeNode, max_path_sum

    root = TreeNode(5)

    assert max_path_sum(root) == 5

def test_max_path_sum_negative():
    from template import TreeNode, max_path_sum

    # Create tree [-3]
    root = TreeNode(-3)

    assert max_path_sum(root) == -3

def test_max_path_sum_all_negative():
    from template import TreeNode, max_path_sum

    # Create tree [-2,-1]
    root = TreeNode(-2, TreeNode(-1))

    assert max_path_sum(root) == -1

def test_max_path_sum_complex():
    from template import TreeNode, max_path_sum

    # Create tree [5,4,8,11,null,13,4,7,2,null,null,null,1]
    root = TreeNode(5)
    root.left = TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2)))
    root.right = TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1)))

    assert max_path_sum(root) == 48
