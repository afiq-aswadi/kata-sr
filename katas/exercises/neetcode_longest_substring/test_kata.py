"""Tests for Longest Substring Without Repeating Characters kata."""

def test_length_of_longest_substring_example1():
    from template import length_of_longest_substring
    assert length_of_longest_substring("abcabcbb") == 3

def test_length_of_longest_substring_example2():
    from template import length_of_longest_substring
    assert length_of_longest_substring("bbbbb") == 1

def test_length_of_longest_substring_example3():
    from template import length_of_longest_substring
    assert length_of_longest_substring("pwwkew") == 3

def test_length_of_longest_substring_empty():
    from template import length_of_longest_substring
    assert length_of_longest_substring("") == 0
