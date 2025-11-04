"""Reference implementation for DFS/BFS kata."""

from collections import deque


def dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """Depth-first search traversal.

    Args:
        graph: adjacency list representation {node: [neighbors]}
        start: starting node

    Returns:
        list of nodes in DFS traversal order
    """
    visited = set()
    result = []

    def visit(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for neighbor in graph.get(node, []):
            visit(neighbor)

    visit(start)
    return result


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """Breadth-first search traversal.

    Args:
        graph: adjacency list representation {node: [neighbors]}
        start: starting node

    Returns:
        list of nodes in BFS traversal order
    """
    visited = {start}
    result = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


def has_path(graph: dict[int, list[int]], start: int, end: int) -> bool:
    """Check if there's a path from start to end node.

    Args:
        graph: adjacency list representation
        start: starting node
        end: target node

    Returns:
        True if path exists, False otherwise
    """
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == end:
            return True

        if node in visited:
            continue

        visited.add(node)
        for neighbor in graph.get(node, []):
            queue.append(neighbor)

    return False
