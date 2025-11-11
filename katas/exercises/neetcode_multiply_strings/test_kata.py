"""Tests for Multiply Strings kata."""

try:
    from user_kata import multiply
except ImportError:
    from .reference import multiply


def test_multiply_example1():
    assert multiply("2", "3") == "6"

def test_multiply_example2():
    assert multiply("123", "456") == "56088"

def test_multiply_zero():
    assert multiply("0", "123") == "0"

def test_multiply_one():
    assert multiply("1", "123") == "123"

def test_multiply_large():
    assert multiply("999", "999") == "998001"
