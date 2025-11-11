"""Tests for Remove Nth Node From End of List kata."""

try:
    from user_kata import ListNode, remove_nth_from_end
except ImportError:
    from .reference import ListNode, remove_nth_from_end


def test_remove_nth_middle():

    # Create list 1->2->3->4->5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    result = remove_nth_from_end(head, 2)

    # Check result: 1->2->3->5
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 2, 3, 5]

def test_remove_nth_single_node():

    head = ListNode(1)
    result = remove_nth_from_end(head, 1)
    assert result is None

def test_remove_nth_last_node():

    # Create list 1->2
    head = ListNode(1, ListNode(2))
    result = remove_nth_from_end(head, 1)

    # Check result: 1
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1]

def test_remove_nth_first_node():

    # Create list 1->2
    head = ListNode(1, ListNode(2))
    result = remove_nth_from_end(head, 2)

    # Check result: 2
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [2]
