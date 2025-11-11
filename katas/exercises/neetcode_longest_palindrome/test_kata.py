"""Tests for Longest Palindromic Substring kata."""

try:
    from user_kata import longest_palindrome
except ImportError:
    from .reference import longest_palindrome


def test_longest_palindrome_example1():
    result = longest_palindrome("babad")
    assert result in ["bab", "aba"]

def test_longest_palindrome_example2():
    assert longest_palindrome("cbbd") == "bb"

def test_longest_palindrome_single():
    assert longest_palindrome("a") == "a"

def test_longest_palindrome_all_same():
    assert longest_palindrome("aaaa") == "aaaa"
