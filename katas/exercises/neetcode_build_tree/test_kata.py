"""Tests for Construct Binary Tree from Preorder and Inorder Traversal kata."""

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

def test_build_tree_example1():
    from template import build_tree

    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]

    root = build_tree(preorder, inorder)
    assert tree_to_list(root) == [3, 9, 20, None, None, 15, 7]

def test_build_tree_example2():
    from template import build_tree

    preorder = [-1]
    inorder = [-1]

    root = build_tree(preorder, inorder)
    assert tree_to_list(root) == [-1]

def test_build_tree_left_skewed():
    from template import build_tree

    preorder = [1, 2, 3]
    inorder = [3, 2, 1]

    root = build_tree(preorder, inorder)
    assert tree_to_list(root) == [1, 2, None, 3]

def test_build_tree_right_skewed():
    from template import build_tree

    preorder = [1, 2, 3]
    inorder = [1, 2, 3]

    root = build_tree(preorder, inorder)
    assert tree_to_list(root) == [1, None, 2, None, 3]

def test_build_tree_balanced():
    from template import build_tree

    preorder = [1, 2, 4, 5, 3, 6, 7]
    inorder = [4, 2, 5, 1, 6, 3, 7]

    root = build_tree(preorder, inorder)
    assert tree_to_list(root) == [1, 2, 3, 4, 5, 6, 7]
