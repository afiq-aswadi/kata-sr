"""Tests for Longest Palindromic Substring kata."""

def test_longest_palindrome_example1():
    from template import longest_palindrome
    result = longest_palindrome("babad")
    assert result in ["bab", "aba"]

def test_longest_palindrome_example2():
    from template import longest_palindrome
    assert longest_palindrome("cbbd") == "bb"

def test_longest_palindrome_single():
    from template import longest_palindrome
    assert longest_palindrome("a") == "a"

def test_longest_palindrome_all_same():
    from template import longest_palindrome
    assert longest_palindrome("aaaa") == "aaaa"
