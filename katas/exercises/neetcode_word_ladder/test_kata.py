"""Tests for Word Ladder kata."""

def test_ladder_length_example1():
    from template import ladder_length
    assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5

def test_ladder_length_example2():
    from template import ladder_length
    assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log"]) == 0

def test_ladder_length_simple():
    from template import ladder_length
    assert ladder_length("hot", "dog", ["hot","dog"]) == 0

def test_ladder_length_direct():
    from template import ladder_length
    assert ladder_length("hot", "dot", ["hot","dot"]) == 2

def test_ladder_length_longer():
    from template import ladder_length
    assert ladder_length("a", "c", ["a","b","c"]) == 2
