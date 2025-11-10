"""Tests for Unique Paths kata."""

def test_unique_paths_example1():
    from template import unique_paths
    assert unique_paths(3, 7) == 28

def test_unique_paths_example2():
    from template import unique_paths
    assert unique_paths(3, 2) == 3

def test_unique_paths_single_row():
    from template import unique_paths
    assert unique_paths(1, 10) == 1

def test_unique_paths_single_col():
    from template import unique_paths
    assert unique_paths(10, 1) == 1

def test_unique_paths_square():
    from template import unique_paths
    assert unique_paths(3, 3) == 6
