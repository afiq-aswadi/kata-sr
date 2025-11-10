"""Tests for Merge Two Sorted Lists kata."""

def test_merge_two_lists_example1():
    from template import ListNode, merge_two_lists

    list1 = ListNode(1, ListNode(2, ListNode(4)))
    list2 = ListNode(1, ListNode(3, ListNode(4)))
    merged = merge_two_lists(list1, list2)

    vals = []
    while merged:
        vals.append(merged.val)
        merged = merged.next
    assert vals == [1, 1, 2, 3, 4, 4]

def test_merge_two_lists_empty():
    from template import merge_two_lists
    assert merge_two_lists(None, None) is None

def test_merge_two_lists_one_empty():
    from template import ListNode, merge_two_lists
    list2 = ListNode(0)
    merged = merge_two_lists(None, list2)
    assert merged.val == 0
