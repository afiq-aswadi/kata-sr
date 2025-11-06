"""LRU Cache kata - reference solution."""

from typing import Any


class Node:
    """Doubly linked list node."""

    def __init__(self, key: int, value: Any):
        self.key = key
        self.value = value
        self.prev: Node | None = None
        self.next: Node | None = None


class LRUCache:
    """LRU Cache implementation."""

    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        self.capacity = capacity
        self.cache: dict[int, Node] = {}

        # Dummy head and tail for easier operations
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> Any:
        """Get value for key, return -1 if not found."""
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # Move to front (most recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: int, value: Any) -> None:
        """Put key-value pair in cache."""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict LRU (before tail)
                lru = self.tail.prev
                assert lru is not None
                self._remove(lru)
                del self.cache[lru.key]

            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
        assert node.prev is not None
        assert node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add node to front (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        assert self.head.next is not None
        self.head.next.prev = node
        self.head.next = node
