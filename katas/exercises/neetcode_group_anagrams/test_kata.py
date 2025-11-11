"""Tests for Group Anagrams kata."""

try:
    from user_kata import group_anagrams
except ImportError:
    from .reference import group_anagrams


def test_group_anagrams_example1():
    result = group_anagrams(["eat","tea","tan","ate","nat","bat"])
    result = [sorted(group) for group in result]
    result = sorted(result)
    expected = [["ate","eat","tea"], ["bat"], ["nat","tan"]]
    expected = [sorted(group) for group in expected]
    expected = sorted(expected)
    assert result == expected

def test_group_anagrams_empty():
    assert group_anagrams([""]) == [[""]]

def test_group_anagrams_single():
    assert group_anagrams(["a"]) == [["a"]]
