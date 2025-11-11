"""Tests for Word Ladder kata."""

try:
    from user_kata import ladder_length
except ImportError:
    from .reference import ladder_length


def test_ladder_length_example1():
    assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5

def test_ladder_length_example2():
    assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log"]) == 0

def test_ladder_length_simple():
    assert ladder_length("hot", "dog", ["hot","dog"]) == 0

def test_ladder_length_direct():
    assert ladder_length("hot", "dot", ["hot","dot"]) == 2

def test_ladder_length_longer():
    assert ladder_length("a", "c", ["a","b","c"]) == 2
