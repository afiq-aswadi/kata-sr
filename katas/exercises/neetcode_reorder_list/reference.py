"""Reorder List - LeetCode 143 - Reference Solution"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reorder_list(head: ListNode | None) -> None:
    if not head or not head.next:
        return

    # Find middle
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    second = slow.next
    slow.next = None
    prev = None
    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp

    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
