"""Cheapest Flights Within K Stops - LeetCode 787 - Reference Solution"""

from collections import defaultdict, deque

def find_cheapest_price(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # BFS with queue: (city, cost, stops)
    queue = deque([(src, 0, 0)])
    min_cost = {src: 0}

    while queue:
        city, cost, stops = queue.popleft()

        if stops > k:
            continue

        for neighbor, price in graph[city]:
            new_cost = cost + price

            # Only proceed if we found a cheaper path or haven't visited
            if neighbor not in min_cost or new_cost < min_cost[neighbor]:
                min_cost[neighbor] = new_cost
                queue.append((neighbor, new_cost, stops + 1))

    return min_cost.get(dst, -1)
