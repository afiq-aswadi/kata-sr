"""Tests for Linked List Cycle kata."""

def test_has_cycle_with_cycle():
    from template import ListNode, has_cycle

    node1 = ListNode(3)
    node2 = ListNode(2)
    node3 = ListNode(0)
    node4 = ListNode(-4)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node2  # cycle

    assert has_cycle(node1) == True

def test_has_cycle_no_cycle():
    from template import ListNode, has_cycle

    node1 = ListNode(1)
    node2 = ListNode(2)
    node1.next = node2

    assert has_cycle(node1) == False

def test_has_cycle_single():
    from template import ListNode, has_cycle
    assert has_cycle(ListNode(1)) == False
