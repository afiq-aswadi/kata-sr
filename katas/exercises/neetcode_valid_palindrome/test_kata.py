"""Tests for Valid Palindrome kata."""

def test_is_palindrome_example1():
    from template import is_palindrome
    assert is_palindrome("A man, a plan, a canal: Panama") == True

def test_is_palindrome_example2():
    from template import is_palindrome
    assert is_palindrome("race a car") == False

def test_is_palindrome_empty():
    from template import is_palindrome
    assert is_palindrome(" ") == True

def test_is_palindrome_single():
    from template import is_palindrome
    assert is_palindrome("a") == True
