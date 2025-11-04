"""DFS/BFS implementation kata."""

from collections import deque


def dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """Depth-first search traversal.

    Args:
        graph: adjacency list representation {node: [neighbors]}
        start: starting node

    Returns:
        list of nodes in DFS traversal order
    """
    # TODO: implement DFS (recursive or iterative with stack)
    # BLANK_START
    pass
    # BLANK_END


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """Breadth-first search traversal.

    Args:
        graph: adjacency list representation {node: [neighbors]}
        start: starting node

    Returns:
        list of nodes in BFS traversal order
    """
    # TODO: implement BFS using a queue
    # BLANK_START
    pass
    # BLANK_END


def has_path(graph: dict[int, list[int]], start: int, end: int) -> bool:
    """Check if there's a path from start to end node.

    Args:
        graph: adjacency list representation
        start: starting node
        end: target node

    Returns:
        True if path exists, False otherwise
    """
    # TODO: use BFS or DFS to check for path
    # BLANK_START
    pass
    # BLANK_END
