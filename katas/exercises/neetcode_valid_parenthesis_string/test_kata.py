"""Tests for Valid Parenthesis String kata."""

def test_check_valid_string_example1():
    from template import check_valid_string
    assert check_valid_string("()") == True

def test_check_valid_string_example2():
    from template import check_valid_string
    assert check_valid_string("(*)") == True

def test_check_valid_string_example3():
    from template import check_valid_string
    assert check_valid_string("(*))") == True

def test_check_valid_string_invalid():
    from template import check_valid_string
    assert check_valid_string("(()") == False

def test_check_valid_string_stars():
    from template import check_valid_string
    assert check_valid_string("(((*") == True
