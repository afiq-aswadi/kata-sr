"""Tests for Longest Repeating Character Replacement kata."""

try:
    from user_kata import character_replacement
except ImportError:
    from .reference import character_replacement


def test_character_replacement_example1():
    assert character_replacement("ABAB", 2) == 4

def test_character_replacement_example2():
    assert character_replacement("AABABBA", 1) == 4

def test_character_replacement_all_same():
    assert character_replacement("AAAA", 0) == 4
