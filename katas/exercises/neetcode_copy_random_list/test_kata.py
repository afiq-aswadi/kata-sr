"""Tests for Copy List with Random Pointer kata."""

try:
    from user_kata import Node, copy_random_list
    from user_kata import copy_random_list
except ImportError:
    from .reference import Node, copy_random_list
    from .reference import copy_random_list


def test_copy_random_list_basic():

    # Create list [[7,null],[13,0],[11,4],[10,2],[1,0]]
    node1 = Node(7)
    node2 = Node(13)
    node3 = Node(11)
    node4 = Node(10)
    node5 = Node(1)

    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5

    node1.random = None
    node2.random = node1
    node3.random = node5
    node4.random = node3
    node5.random = node1

    # Copy the list
    copied_head = copy_random_list(node1)

    # Verify values
    vals = []
    current = copied_head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [7, 13, 11, 10, 1]

    # Verify it's a deep copy (different objects)
    assert copied_head is not node1

    # Verify random pointers
    current_orig = node1
    current_copy = copied_head
    nodes_orig = []
    nodes_copy = []
    while current_orig:
        nodes_orig.append(current_orig)
        nodes_copy.append(current_copy)
        current_orig = current_orig.next
        current_copy = current_copy.next

    # Check random pointers maintain same relative positions
    for i, (orig, copy) in enumerate(zip(nodes_orig, nodes_copy)):
        if orig.random is None:
            assert copy.random is None
        else:
            orig_idx = nodes_orig.index(orig.random)
            assert copy.random is nodes_copy[orig_idx]

def test_copy_random_list_empty():
    assert copy_random_list(None) is None

def test_copy_random_list_single():

    node = Node(1)
    node.random = node

    copied = copy_random_list(node)
    assert copied.val == 1
    assert copied.random is copied
    assert copied is not node
