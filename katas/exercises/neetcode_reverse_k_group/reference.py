"""Reverse Nodes in k-Group - LeetCode 25 - Reference Solution"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None:
    # Check if there are k nodes to reverse
    count = 0
    current = head
    while current and count < k:
        current = current.next
        count += 1

    if count < k:
        return head

    # Reverse k nodes
    prev = None
    current = head
    for _ in range(k):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    # Recursively reverse remaining nodes
    head.next = reverse_k_group(current, k)

    return prev
