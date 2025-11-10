"""Merge K Sorted Lists - LeetCode 23 - Reference Solution"""

import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    if not lists:
        return None

    # Min heap: (value, index, node)
    # Index is used to break ties for nodes with same value
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
