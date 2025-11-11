"""Tests for Linked List Cycle kata."""

try:
    from user_kata import ListNode, has_cycle
except ImportError:
    from .reference import ListNode, has_cycle


def test_has_cycle_with_cycle():

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

    node1 = ListNode(1)
    node2 = ListNode(2)
    node1.next = node2

    assert has_cycle(node1) == False

def test_has_cycle_single():
    assert has_cycle(ListNode(1)) == False
