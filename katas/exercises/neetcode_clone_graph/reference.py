"""Clone Graph - LeetCode 133 - Reference Solution"""

class Node:
    def __init__(self, val: int = 0, neighbors: list['Node'] | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Node | None) -> Node | None:
    if not node:
        return None

    old_to_new = {}

    def dfs(node: Node) -> Node:
        if node in old_to_new:
            return old_to_new[node]

        copy = Node(node.val)
        old_to_new[node] = copy

        for neighbor in node.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)
