"""Swim in Rising Water - LeetCode 778 - Reference Solution"""

import heapq

def swim_in_water(grid: list[list[int]]) -> int:
    n = len(grid)
    visited = set()
    min_heap = [(grid[0][0], 0, 0)]  # (max_elevation, row, col)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while min_heap:
        max_elevation, r, c = heapq.heappop(min_heap)

        if r == n - 1 and c == n - 1:
            return max_elevation

        if (r, c) in visited:
            continue

        visited.add((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and
                (nr, nc) not in visited):
                heapq.heappush(min_heap, (max(max_elevation, grid[nr][nc]), nr, nc))

    return -1
