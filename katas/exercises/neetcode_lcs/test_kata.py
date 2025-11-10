"""Tests for Longest Common Subsequence kata."""

def test_lcs_example1():
    from template import longest_common_subsequence
    assert longest_common_subsequence("abcde", "ace") == 3

def test_lcs_example2():
    from template import longest_common_subsequence
    assert longest_common_subsequence("abc", "abc") == 3

def test_lcs_example3():
    from template import longest_common_subsequence
    assert longest_common_subsequence("abc", "def") == 0

def test_lcs_empty():
    from template import longest_common_subsequence
    assert longest_common_subsequence("", "abc") == 0

def test_lcs_partial():
    from template import longest_common_subsequence
    assert longest_common_subsequence("abc", "abcd") == 3
