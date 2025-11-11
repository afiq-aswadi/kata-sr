"""Tests for Valid Anagram kata."""

try:
    from user_kata import is_anagram
except ImportError:
    from .reference import is_anagram


def test_is_anagram_example1():
    assert is_anagram("anagram", "nagaram") == True

def test_is_anagram_example2():
    assert is_anagram("rat", "car") == False

def test_is_anagram_empty():
    assert is_anagram("", "") == True
