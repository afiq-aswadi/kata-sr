"""LRU Cache - LeetCode 146 - Reference Solution"""

class Node:
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        # Doubly linked list with dummy head and tail
        self.left = Node()  # LRU (dummy head)
        self.right = Node()  # MRU (dummy tail)
        self.left.next = self.right
        self.right.prev = self.left

    def remove(self, node: Node) -> None:
        """Remove node from list"""
        prev, next = node.prev, node.next
        prev.next = next
        next.prev = prev

    def insert(self, node: Node) -> None:
        """Insert node at right (most recently used)"""
        prev = self.right.prev
        prev.next = node
        node.prev = prev
        node.next = self.right
        self.right.prev = node

    def get(self, key: int) -> int:
        if key in self.cache:
            # Move to most recently used
            node = self.cache[key]
            self.remove(node)
            self.insert(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing key
            self.remove(self.cache[key])

        node = Node(key, value)
        self.cache[key] = node
        self.insert(node)

        if len(self.cache) > self.capacity:
            # Remove least recently used (leftmost)
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
