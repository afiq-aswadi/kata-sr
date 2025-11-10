"""Tests for Clone Graph kata."""

def build_graph(adj_list: list[list[int]]):
    """Helper to build graph from adjacency list."""
    from template import Node

    if not adj_list:
        return None

    nodes = {i + 1: Node(i + 1) for i in range(len(adj_list))}

    for i, neighbors in enumerate(adj_list):
        node = nodes[i + 1]
        node.neighbors = [nodes[n] for n in neighbors]

    return nodes[1] if nodes else None

def graph_to_adj_list(node):
    """Helper to convert graph back to adjacency list."""
    if not node:
        return []

    visited = {}

    def dfs(n):
        if n.val in visited:
            return
        visited[n.val] = [neighbor.val for neighbor in n.neighbors]
        for neighbor in n.neighbors:
            dfs(neighbor)

    dfs(node)

    if not visited:
        return []

    max_val = max(visited.keys())
    result = [[] for _ in range(max_val)]
    for val in range(1, max_val + 1):
        if val in visited:
            result[val - 1] = visited[val]

    return result

def test_clone_graph_example1():
    from template import clone_graph
    node = build_graph([[2,4],[1,3],[2,4],[1,3]])
    cloned = clone_graph(node)
    assert graph_to_adj_list(cloned) == [[2,4],[1,3],[2,4],[1,3]]
    assert cloned is not node

def test_clone_graph_example2():
    from template import clone_graph
    node = build_graph([[]])
    cloned = clone_graph(node)
    assert graph_to_adj_list(cloned) == [[]]
    assert cloned is not node

def test_clone_graph_example3():
    from template import clone_graph
    node = build_graph([])
    cloned = clone_graph(node)
    assert cloned is None

def test_clone_graph_simple():
    from template import clone_graph
    node = build_graph([[2],[1]])
    cloned = clone_graph(node)
    assert graph_to_adj_list(cloned) == [[2],[1]]
    assert cloned is not node
