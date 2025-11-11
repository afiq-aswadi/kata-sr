"""Tests for Reverse Linked List kata."""

try:
    from user_kata import ListNode, reverse_list
    from user_kata import reverse_list
except ImportError:
    from .reference import ListNode, reverse_list
    from .reference import reverse_list


def test_reverse_list_multiple():

    # Create list 1->2->3->4->5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    reversed_head = reverse_list(head)

    # Check reversed list
    vals = []
    current = reversed_head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [5, 4, 3, 2, 1]

def test_reverse_list_empty():
    assert reverse_list(None) is None

def test_reverse_list_single():
    head = ListNode(1)
    reversed_head = reverse_list(head)
    assert reversed_head.val == 1
    assert reversed_head.next is None
