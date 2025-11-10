"""Tests for Valid Anagram kata."""

def test_is_anagram_example1():
    from template import is_anagram
    assert is_anagram("anagram", "nagaram") == True

def test_is_anagram_example2():
    from template import is_anagram
    assert is_anagram("rat", "car") == False

def test_is_anagram_empty():
    from template import is_anagram
    assert is_anagram("", "") == True
