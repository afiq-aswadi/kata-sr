"""Reconstruct Itinerary - LeetCode 332 - Reference Solution"""

from collections import defaultdict

def find_itinerary(tickets: list[list[str]]) -> list[str]:
    # Build adjacency list with sorted destinations
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    route = []

    def dfs(airport: str) -> None:
        while graph[airport]:
            next_dest = graph[airport].pop()
            dfs(next_dest)
        route.append(airport)

    dfs("JFK")

    return route[::-1]
