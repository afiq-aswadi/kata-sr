"""Walls and Gates - LeetCode 286 - Reference Solution"""

from collections import deque

def walls_and_gates(rooms: list[list[int]]) -> None:
    """
    Do not return anything, modify rooms in-place instead.
    """
    if not rooms:
        return

    rows, cols = len(rooms), len(rooms[0])
    queue = deque()
    INF = 2147483647

    # Add all gates to queue
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # BFS from all gates
    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                rooms[nr][nc] == INF):
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))
