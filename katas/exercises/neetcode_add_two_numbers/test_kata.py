"""Tests for Add Two Numbers kata."""

def test_add_two_numbers_basic():
    from template import ListNode, add_two_numbers

    # Create l1 = [2,4,3] (342)
    l1 = ListNode(2, ListNode(4, ListNode(3)))
    # Create l2 = [5,6,4] (465)
    l2 = ListNode(5, ListNode(6, ListNode(4)))

    result = add_two_numbers(l1, l2)

    # Check result: [7,0,8] (807)
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [7, 0, 8]

def test_add_two_numbers_zeros():
    from template import ListNode, add_two_numbers

    l1 = ListNode(0)
    l2 = ListNode(0)

    result = add_two_numbers(l1, l2)

    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [0]

def test_add_two_numbers_different_lengths():
    from template import ListNode, add_two_numbers

    # Create l1 = [9,9,9,9,9,9,9]
    l1 = ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9)))))))
    # Create l2 = [9,9,9,9]
    l2 = ListNode(9, ListNode(9, ListNode(9, ListNode(9))))

    result = add_two_numbers(l1, l2)

    # Check result: [8,9,9,9,0,0,0,1]
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [8, 9, 9, 9, 0, 0, 0, 1]

def test_add_two_numbers_carry():
    from template import ListNode, add_two_numbers

    # Create l1 = [9,9]
    l1 = ListNode(9, ListNode(9))
    # Create l2 = [1]
    l2 = ListNode(1)

    result = add_two_numbers(l1, l2)

    # Check result: [0,0,1] (99 + 1 = 100)
    vals = []
    current = result
    while current:
        vals.append(current.val)
        current = current.next
    assert vals == [0, 0, 1]
