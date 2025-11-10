"""Tests for Invert Binary Tree kata."""

def tree_to_list(root):
    """Helper to convert tree to level-order list."""
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()
    return result

def test_invert_tree_example1():
    from template import TreeNode, invert_tree

    # Create tree [4,2,7,1,3,6,9]
    root = TreeNode(4)
    root.left = TreeNode(2, TreeNode(1), TreeNode(3))
    root.right = TreeNode(7, TreeNode(6), TreeNode(9))

    inverted = invert_tree(root)
    assert tree_to_list(inverted) == [4, 7, 2, 9, 6, 3, 1]

def test_invert_tree_example2():
    from template import TreeNode, invert_tree

    # Create tree [2,1,3]
    root = TreeNode(2, TreeNode(1), TreeNode(3))

    inverted = invert_tree(root)
    assert tree_to_list(inverted) == [2, 3, 1]

def test_invert_tree_empty():
    from template import invert_tree
    assert invert_tree(None) is None

def test_invert_tree_single():
    from template import TreeNode, invert_tree
    root = TreeNode(1)
    inverted = invert_tree(root)
    assert tree_to_list(inverted) == [1]
