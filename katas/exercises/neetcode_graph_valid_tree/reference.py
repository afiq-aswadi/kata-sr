"""Graph Valid Tree - LeetCode 261 - Reference Solution"""

def valid_tree(n: int, edges: list[list[int]]) -> bool:
    # A valid tree must have exactly n-1 edges and be fully connected
    if len(edges) != n - 1:
        return False

    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)

    # DFS to check if all nodes are reachable from node 0
    visited = set()

    def dfs(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        for neighbor in adj[node]:
            dfs(neighbor)

    dfs(0)

    return len(visited) == n
