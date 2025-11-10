"""Tests for Edit Distance kata."""

def test_min_distance_example1():
    from template import min_distance
    assert min_distance("horse", "ros") == 3

def test_min_distance_example2():
    from template import min_distance
    assert min_distance("intention", "execution") == 5

def test_min_distance_empty():
    from template import min_distance
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3

def test_min_distance_same():
    from template import min_distance
    assert min_distance("abc", "abc") == 0
