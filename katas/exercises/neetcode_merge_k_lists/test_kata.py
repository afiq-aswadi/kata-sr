"""Tests for Merge K Sorted Lists kata."""

def test_merge_k_lists_basic():
    from template import ListNode, merge_k_lists

    # Create lists [[1,4,5],[1,3,4],[2,6]]
    list1 = ListNode(1, ListNode(4, ListNode(5)))
    list2 = ListNode(1, ListNode(3, ListNode(4)))
    list3 = ListNode(2, ListNode(6))

    result = merge_k_lists([list1, list2, list3])

    # Check result: [1,1,2,3,4,4,5,6]
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 1, 2, 3, 4, 4, 5, 6]

def test_merge_k_lists_empty():
    from template import merge_k_lists
    assert merge_k_lists([]) is None

def test_merge_k_lists_empty_lists():
    from template import merge_k_lists
    assert merge_k_lists([None, None]) is None

def test_merge_k_lists_single():
    from template import ListNode, merge_k_lists

    list1 = ListNode(1, ListNode(2, ListNode(3)))
    result = merge_k_lists([list1])

    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 2, 3]

def test_merge_k_lists_mixed_empty():
    from template import ListNode, merge_k_lists

    list1 = ListNode(1)
    list2 = None
    list3 = ListNode(2)

    result = merge_k_lists([list1, list2, list3])

    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [1, 2]
