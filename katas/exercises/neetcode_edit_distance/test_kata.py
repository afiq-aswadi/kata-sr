"""Tests for Edit Distance kata."""

try:
    from user_kata import min_distance
except ImportError:
    from .reference import min_distance


def test_min_distance_example1():
    assert min_distance("horse", "ros") == 3

def test_min_distance_example2():
    assert min_distance("intention", "execution") == 5

def test_min_distance_empty():
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3

def test_min_distance_same():
    assert min_distance("abc", "abc") == 0
