"""Tests for Reorder List kata."""

try:
    from user_kata import ListNode, reorder_list
except ImportError:
    from .reference import ListNode, reorder_list


def test_reorder_list_even():

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 4, 2, 3]

def test_reorder_list_odd():

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 5, 2, 4, 3]
