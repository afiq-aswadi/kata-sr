"""LRU Cache kata."""

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
        """Initialize LRU cache with given capacity.

        Args:
            capacity: maximum number of items
        """
        # TODO: initialize hash map, dummy head/tail nodes, capacity
        # BLANK_START
        pass
        # BLANK_END

    def get(self, key: int) -> Any:
        """Get value for key, return -1 if not found.

        Args:
            key: key to lookup

        Returns:
            value if found, -1 otherwise
        """
        # TODO: lookup in hash map, move to front (most recent)
        # BLANK_START
        pass
        # BLANK_END

    def put(self, key: int, value: Any) -> None:
        """Put key-value pair in cache.

        Args:
            key: key to store
            value: value to store
        """
        # TODO: update if exists, otherwise insert
        # If at capacity, evict LRU item
        # BLANK_START
        pass
        # BLANK_END

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
        # TODO: update prev and next pointers
        # BLANK_START
        pass
        # BLANK_END

    def _add_to_front(self, node: Node) -> None:
        """Add node to front (most recently used)."""
        # TODO: insert after dummy head
        # BLANK_START
        pass
        # BLANK_END
