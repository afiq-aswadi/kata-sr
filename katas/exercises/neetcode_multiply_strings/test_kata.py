"""Tests for Multiply Strings kata."""

def test_multiply_example1():
    from template import multiply
    assert multiply("2", "3") == "6"

def test_multiply_example2():
    from template import multiply
    assert multiply("123", "456") == "56088"

def test_multiply_zero():
    from template import multiply
    assert multiply("0", "123") == "0"

def test_multiply_one():
    from template import multiply
    assert multiply("1", "123") == "123"

def test_multiply_large():
    from template import multiply
    assert multiply("999", "999") == "998001"
