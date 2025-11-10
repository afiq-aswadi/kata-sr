"""Min Cost to Connect All Points - LeetCode 1584 - Reference Solution"""

import heapq

def min_cost_connect_points(points: list[list[int]]) -> int:
    n = len(points)

    # Prim's algorithm
    visited = set()
    min_heap = [(0, 0)]  # (cost, point_index)
    total_cost = 0

    while len(visited) < n:
        cost, i = heapq.heappop(min_heap)

        if i in visited:
            continue

        visited.add(i)
        total_cost += cost

        # Add all edges from this point to unvisited points
        for j in range(n):
            if j not in visited:
                manhattan_dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                heapq.heappush(min_heap, (manhattan_dist, j))

    return total_cost
