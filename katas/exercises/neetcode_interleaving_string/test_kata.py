"""Tests for Interleaving String kata."""

try:
    from user_kata import is_interleave
except ImportError:
    from .reference import is_interleave


def test_is_interleave_example1():
    assert is_interleave("aabcc", "dbbca", "aadbbcbcac") == True

def test_is_interleave_example2():
    assert is_interleave("aabcc", "dbbca", "aadbbbaccc") == False

def test_is_interleave_example3():
    assert is_interleave("", "", "") == True

def test_is_interleave_empty():
    assert is_interleave("", "abc", "abc") == True
    assert is_interleave("abc", "", "abc") == True
