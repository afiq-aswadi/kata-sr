"""Tests for Valid Parentheses kata."""

try:
    from user_kata import is_valid
except ImportError:
    from .reference import is_valid


def test_is_valid_simple():
    assert is_valid("()") == True

def test_is_valid_multiple():
    assert is_valid("()[]{}") == True

def test_is_valid_invalid():
    assert is_valid("(]") == False

def test_is_valid_nested():
    assert is_valid("{[]}") == True
