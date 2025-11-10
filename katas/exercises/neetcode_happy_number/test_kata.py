"""Tests for Happy Number kata."""

def test_is_happy_example1():
    from template import is_happy
    assert is_happy(19) == True

def test_is_happy_example2():
    from template import is_happy
    assert is_happy(2) == False

def test_is_happy_one():
    from template import is_happy
    assert is_happy(1) == True

def test_is_happy_seven():
    from template import is_happy
    assert is_happy(7) == True

def test_is_happy_not_happy():
    from template import is_happy
    assert is_happy(4) == False
