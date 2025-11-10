"""Network Delay Time - LeetCode 743 - Reference Solution"""

import heapq
from collections import defaultdict

def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra's algorithm
    min_heap = [(0, k)]  # (time, node)
    visited = set()
    max_time = 0

    while min_heap:
        time, node = heapq.heappop(min_heap)

        if node in visited:
            continue

        visited.add(node)
        max_time = max(max_time, time)

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (time + weight, neighbor))

    return max_time if len(visited) == n else -1
