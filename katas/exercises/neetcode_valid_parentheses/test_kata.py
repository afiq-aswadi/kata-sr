"""Tests for Valid Parentheses kata."""

def test_is_valid_simple():
    from template import is_valid
    assert is_valid("()") == True

def test_is_valid_multiple():
    from template import is_valid
    assert is_valid("()[]{}") == True

def test_is_valid_invalid():
    from template import is_valid
    assert is_valid("(]") == False

def test_is_valid_nested():
    from template import is_valid
    assert is_valid("{[]}") == True
