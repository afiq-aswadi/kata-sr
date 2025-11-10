"""Tests for Interleaving String kata."""

def test_is_interleave_example1():
    from template import is_interleave
    assert is_interleave("aabcc", "dbbca", "aadbbcbcac") == True

def test_is_interleave_example2():
    from template import is_interleave
    assert is_interleave("aabcc", "dbbca", "aadbbbaccc") == False

def test_is_interleave_example3():
    from template import is_interleave
    assert is_interleave("", "", "") == True

def test_is_interleave_empty():
    from template import is_interleave
    assert is_interleave("", "abc", "abc") == True
    assert is_interleave("abc", "", "abc") == True
