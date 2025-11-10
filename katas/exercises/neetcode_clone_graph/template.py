"""Clone Graph - LeetCode 133"""

class Node:
    def __init__(self, val: int = 0, neighbors: list['Node'] | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Node | None) -> Node | None:
    # TODO: Use DFS/BFS with a hashmap to track old->new node mapping
    # BLANK_START
    pass
    # BLANK_END
