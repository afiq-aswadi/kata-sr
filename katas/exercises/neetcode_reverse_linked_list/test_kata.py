"""Tests for Reverse Linked List kata."""

def test_reverse_list_multiple():
    from template import ListNode, reverse_list

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
    from template import reverse_list
    assert reverse_list(None) is None

def test_reverse_list_single():
    from template import ListNode, reverse_list
    head = ListNode(1)
    reversed_head = reverse_list(head)
    assert reversed_head.val == 1
    assert reversed_head.next is None
