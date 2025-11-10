"""Tests for Longest Repeating Character Replacement kata."""

def test_character_replacement_example1():
    from template import character_replacement
    assert character_replacement("ABAB", 2) == 4

def test_character_replacement_example2():
    from template import character_replacement
    assert character_replacement("AABABBA", 1) == 4

def test_character_replacement_all_same():
    from template import character_replacement
    assert character_replacement("AAAA", 0) == 4
