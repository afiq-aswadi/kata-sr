"""Tests for Reverse Nodes in k-Group kata."""

try:
    from user_kata import ListNode, reverse_k_group
except ImportError:
    from .reference import ListNode, reverse_k_group


def test_reverse_k_group_k2():

    # Create list 1->2->3->4->5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    result = reverse_k_group(head, 2)

    # Check result: 2->1->4->3->5
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [2, 1, 4, 3, 5]

def test_reverse_k_group_k3():

    # Create list 1->2->3->4->5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    result = reverse_k_group(head, 3)

    # Check result: 3->2->1->4->5
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [3, 2, 1, 4, 5]

def test_reverse_k_group_k1():

    # Create list 1->2->3
    head = ListNode(1, ListNode(2, ListNode(3)))
    result = reverse_k_group(head, 1)

    # Check result: 1->2->3 (no change)
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 2, 3]

def test_reverse_k_group_exact_multiple():

    # Create list 1->2->3->4
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    result = reverse_k_group(head, 2)

    # Check result: 2->1->4->3
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [2, 1, 4, 3]

def test_reverse_k_group_single_node():

    head = ListNode(1)
    result = reverse_k_group(head, 1)

    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1]
