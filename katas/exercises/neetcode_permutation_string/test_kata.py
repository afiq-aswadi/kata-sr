"""Tests for Permutation in String kata."""

def test_check_inclusion_example1():
    from template import check_inclusion
    assert check_inclusion("ab", "eidbaooo") == True

def test_check_inclusion_example2():
    from template import check_inclusion
    assert check_inclusion("ab", "eidboaoo") == False

def test_check_inclusion_exact_match():
    from template import check_inclusion
    assert check_inclusion("abc", "bca") == True
