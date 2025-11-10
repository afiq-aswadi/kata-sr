"""Tests for Number of Connected Components kata."""

def test_count_components_example1():
    from template import count_components
    assert count_components(5, [[0,1],[1,2],[3,4]]) == 2

def test_count_components_example2():
    from template import count_components
    assert count_components(5, [[0,1],[1,2],[2,3],[3,4]]) == 1

def test_count_components_no_edges():
    from template import count_components
    assert count_components(4, []) == 4

def test_count_components_single():
    from template import count_components
    assert count_components(1, []) == 1

def test_count_components_fully_connected():
    from template import count_components
    assert count_components(3, [[0,1],[1,2],[0,2]]) == 1
