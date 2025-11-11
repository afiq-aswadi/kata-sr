"""Tests for Longest Common Subsequence kata."""

try:
    from user_kata import longest_common_subsequence
except ImportError:
    from .reference import longest_common_subsequence


def test_lcs_example1():
    assert longest_common_subsequence("abcde", "ace") == 3

def test_lcs_example2():
    assert longest_common_subsequence("abc", "abc") == 3

def test_lcs_example3():
    assert longest_common_subsequence("abc", "def") == 0

def test_lcs_empty():
    assert longest_common_subsequence("", "abc") == 0

def test_lcs_partial():
    assert longest_common_subsequence("abc", "abcd") == 3
