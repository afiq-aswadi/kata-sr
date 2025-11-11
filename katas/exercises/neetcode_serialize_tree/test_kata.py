"""Tests for Serialize and Deserialize Binary Tree kata."""

try:
    from user_kata import TreeNode, serialize, deserialize
    from user_kata import serialize, deserialize
except ImportError:
    from .reference import TreeNode, serialize, deserialize
    from .reference import serialize, deserialize


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

def test_serialize_deserialize_example1():

    # Create tree [1,2,3,null,null,4,5]
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3, TreeNode(4), TreeNode(5))

    serialized = serialize(root)
    deserialized = deserialize(serialized)

    assert tree_to_list(deserialized) == [1, 2, 3, None, None, 4, 5]

def test_serialize_deserialize_empty():

    serialized = serialize(None)
    deserialized = deserialize(serialized)

    assert deserialized is None

def test_serialize_deserialize_single():

    root = TreeNode(1)

    serialized = serialize(root)
    deserialized = deserialize(serialized)

    assert tree_to_list(deserialized) == [1]

def test_serialize_deserialize_left_skewed():

    # Create tree [1,2,null,3]
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(3)

    serialized = serialize(root)
    deserialized = deserialize(serialized)

    assert tree_to_list(deserialized) == [1, 2, None, 3]

def test_serialize_deserialize_complete():

    # Create tree [1,2,3,4,5,6,7]
    root = TreeNode(1)
    root.left = TreeNode(2, TreeNode(4), TreeNode(5))
    root.right = TreeNode(3, TreeNode(6), TreeNode(7))

    serialized = serialize(root)
    deserialized = deserialize(serialized)

    assert tree_to_list(deserialized) == [1, 2, 3, 4, 5, 6, 7]

def test_serialize_deserialize_negative():

    # Create tree with negative values
    root = TreeNode(-1, TreeNode(-2), TreeNode(-3))

    serialized = serialize(root)
    deserialized = deserialize(serialized)

    assert tree_to_list(deserialized) == [-1, -2, -3]
