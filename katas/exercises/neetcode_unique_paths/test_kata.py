"""Tests for Unique Paths kata."""

try:
    from user_kata import unique_paths
except ImportError:
    from .reference import unique_paths


def test_unique_paths_example1():
    assert unique_paths(3, 7) == 28

def test_unique_paths_example2():
    assert unique_paths(3, 2) == 3

def test_unique_paths_single_row():
    assert unique_paths(1, 10) == 1

def test_unique_paths_single_col():
    assert unique_paths(10, 1) == 1

def test_unique_paths_square():
    assert unique_paths(3, 3) == 6
