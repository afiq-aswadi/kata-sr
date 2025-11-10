"""Number of Connected Components in Undirected Graph - LeetCode 323 - Reference Solution"""

def count_components(n: int, edges: list[list[int]]) -> int:
    parent = [i for i in range(n)]
    rank = [1] * n

    def find(node: int) -> int:
        while node != parent[node]:
            parent[node] = parent[parent[node]]  # Path compression
            node = parent[node]
        return node

    def union(n1: int, n2: int) -> bool:
        p1, p2 = find(n1), find(n2)

        if p1 == p2:
            return False

        if rank[p1] > rank[p2]:
            parent[p2] = p1
            rank[p1] += rank[p2]
        else:
            parent[p1] = p2
            rank[p2] += rank[p1]

        return True

    components = n
    for n1, n2 in edges:
        if union(n1, n2):
            components -= 1

    return components
