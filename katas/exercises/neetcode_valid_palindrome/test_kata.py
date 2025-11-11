"""Tests for Valid Palindrome kata."""

try:
    from user_kata import is_palindrome
except ImportError:
    from .reference import is_palindrome


def test_is_palindrome_example1():
    assert is_palindrome("A man, a plan, a canal: Panama") == True

def test_is_palindrome_example2():
    assert is_palindrome("race a car") == False

def test_is_palindrome_empty():
    assert is_palindrome(" ") == True

def test_is_palindrome_single():
    assert is_palindrome("a") == True
