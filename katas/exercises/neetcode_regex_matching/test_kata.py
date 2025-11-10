"""Tests for Regular Expression Matching kata."""

def test_is_match_example1():
    from template import is_match
    assert is_match("aa", "a") == False

def test_is_match_example2():
    from template import is_match
    assert is_match("aa", "a*") == True

def test_is_match_example3():
    from template import is_match
    assert is_match("ab", ".*") == True

def test_is_match_dot():
    from template import is_match
    assert is_match("ab", "..") == True

def test_is_match_complex():
    from template import is_match
    assert is_match("aab", "c*a*b") == True
