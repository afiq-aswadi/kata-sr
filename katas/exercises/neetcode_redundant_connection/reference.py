"""Redundant Connection - LeetCode 684 - Reference Solution"""

def find_redundant_connection(edges: list[list[int]]) -> list[int]:
    n = len(edges)
    parent = [i for i in range(n + 1)]
    rank = [1] * (n + 1)

    def find(node: int) -> int:
        while node != parent[node]:
            parent[node] = parent[parent[node]]  # Path compression
            node = parent[node]
        return node

    def union(n1: int, n2: int) -> bool:
        p1, p2 = find(n1), find(n2)

        if p1 == p2:
            return False  # Already connected, cycle detected

        if rank[p1] > rank[p2]:
            parent[p2] = p1
            rank[p1] += rank[p2]
        else:
            parent[p1] = p2
            rank[p2] += rank[p1]

        return True

    for n1, n2 in edges:
        if not union(n1, n2):
            return [n1, n2]

    return []
