"""Tests for Regular Expression Matching kata."""

try:
    from user_kata import is_match
except ImportError:
    from .reference import is_match


def test_is_match_example1():
    assert is_match("aa", "a") == False

def test_is_match_example2():
    assert is_match("aa", "a*") == True

def test_is_match_example3():
    assert is_match("ab", ".*") == True

def test_is_match_dot():
    assert is_match("ab", "..") == True

def test_is_match_complex():
    assert is_match("aab", "c*a*b") == True
