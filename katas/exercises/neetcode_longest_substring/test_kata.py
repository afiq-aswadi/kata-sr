"""Tests for Longest Substring Without Repeating Characters kata."""

try:
    from user_kata import length_of_longest_substring
except ImportError:
    from .reference import length_of_longest_substring


def test_length_of_longest_substring_example1():
    assert length_of_longest_substring("abcabcbb") == 3

def test_length_of_longest_substring_example2():
    assert length_of_longest_substring("bbbbb") == 1

def test_length_of_longest_substring_example3():
    assert length_of_longest_substring("pwwkew") == 3

def test_length_of_longest_substring_empty():
    assert length_of_longest_substring("") == 0
