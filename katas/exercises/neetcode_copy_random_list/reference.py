"""Copy List with Random Pointer - LeetCode 138 - Reference Solution"""

class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head: Node | None) -> Node | None:
    if not head:
        return None

    # First pass: create all nodes and store mapping
    old_to_new = {}
    current = head
    while current:
        old_to_new[current] = Node(current.val)
        current = current.next

    # Second pass: connect next and random pointers
    current = head
    while current:
        if current.next:
            old_to_new[current].next = old_to_new[current.next]
        if current.random:
            old_to_new[current].random = old_to_new[current.random]
        current = current.next

    return old_to_new[head]
