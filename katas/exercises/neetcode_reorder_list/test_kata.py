"""Tests for Reorder List kata."""

def test_reorder_list_even():
    from template import ListNode, reorder_list

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 4, 2, 3]

def test_reorder_list_odd():
    from template import ListNode, reorder_list

    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    reorder_list(head)

    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 5, 2, 4, 3]
