"""Tests for DFS/BFS kata."""

from user_kata import bfs, dfs, has_path


def test_dfs_simple_graph():
    """DFS should visit all reachable nodes."""
    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [],
    }
    result = dfs(graph, 1)
    assert len(result) == 4, f"Expected 4 nodes, got {len(result)}"
    assert result[0] == 1, "Should start at node 1"
    assert set(result) == {1, 2, 3, 4}, f"Expected {{1,2,3,4}}, got {set(result)}"


def test_dfs_with_cycle():
    """DFS should handle cycles without infinite loops."""
    graph = {
        1: [2],
        2: [3],
        3: [1, 4],
        4: [],
    }
    result = dfs(graph, 1)
    assert len(result) == 4, "Should visit each node exactly once despite cycle"
    assert set(result) == {1, 2, 3, 4}


def test_dfs_disconnected():
    """DFS should only visit connected component."""
    graph = {
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
    result = dfs(graph, 1)
    assert set(result) == {1, 2}, "Should only visit connected component"


def test_bfs_simple_graph():
    """BFS should visit nodes level by level."""
    graph = {
        1: [2, 3],
        2: [4],
        3: [5],
        4: [],
        5: [],
    }
    result = bfs(graph, 1)
    assert result[0] == 1, "Should start at node 1"
    # BFS visits level by level, so 2 and 3 before 4 and 5
    assert set(result[:3]) == {1, 2, 3}, "First 3 nodes should be 1, 2, 3"
    assert set(result) == {1, 2, 3, 4, 5}


def test_bfs_with_cycle():
    """BFS should handle cycles correctly."""
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [],
        4: [],
    }
    result = bfs(graph, 1)
    assert len(result) == 4, "Should handle cycles correctly"
    assert set(result) == {1, 2, 3, 4}


def test_bfs_single_node():
    """BFS of single node should return just that node."""
    graph = {1: []}
    result = bfs(graph, 1)
    assert result == [1]


def test_has_path_exists():
    """Should find path when it exists."""
    graph = {
        1: [2],
        2: [3],
        3: [4],
        4: [],
    }
    assert has_path(graph, 1, 4) is True
    assert has_path(graph, 1, 3) is True
    assert has_path(graph, 2, 4) is True


def test_has_path_does_not_exist():
    """Should return False when no path exists."""
    graph = {
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
    assert has_path(graph, 1, 4) is False
    assert has_path(graph, 2, 3) is False


def test_has_path_same_node():
    """Path from node to itself should exist."""
    graph = {1: [2], 2: []}
    assert has_path(graph, 1, 1) is True


def test_has_path_with_cycle():
    """Should handle cycles when checking for path."""
    graph = {
        1: [2],
        2: [3],
        3: [1, 4],
        4: [],
    }
    assert has_path(graph, 1, 4) is True
    # shouldn't infinite loop on cycle
    assert has_path(graph, 1, 1) is True
