"""Tests for Group Anagrams kata."""

def test_group_anagrams_example1():
    from template import group_anagrams
    result = group_anagrams(["eat","tea","tan","ate","nat","bat"])
    result = [sorted(group) for group in result]
    result = sorted(result)
    expected = [["ate","eat","tea"], ["bat"], ["nat","tan"]]
    expected = [sorted(group) for group in expected]
    expected = sorted(expected)
    assert result == expected

def test_group_anagrams_empty():
    from template import group_anagrams
    assert group_anagrams([""]) == [[""]]

def test_group_anagrams_single():
    from template import group_anagrams
    assert group_anagrams(["a"]) == [["a"]]
