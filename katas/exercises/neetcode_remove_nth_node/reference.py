"""Remove Nth Node From End of List - LeetCode 19 - Reference Solution"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    dummy = ListNode(0, head)
    left = dummy
    right = head

    # Move right pointer n nodes ahead
    for _ in range(n):
        right = right.next

    # Move both pointers until right reaches the end
    while right:
        left = left.next
        right = right.next

    # Remove the nth node
    left.next = left.next.next

    return dummy.next
