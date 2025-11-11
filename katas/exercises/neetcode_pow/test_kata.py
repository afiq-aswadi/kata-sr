"""Tests for Pow(x, n) kata."""

try:
    from user_kata import my_pow
except ImportError:
    from .reference import my_pow


def test_my_pow_example1():
    assert abs(my_pow(2.0, 10) - 1024.0) < 1e-5

def test_my_pow_example2():
    assert abs(my_pow(2.1, 3) - 9.261) < 1e-5

def test_my_pow_example3():
    assert abs(my_pow(2.0, -2) - 0.25) < 1e-5

def test_my_pow_zero():
    assert abs(my_pow(2.0, 0) - 1.0) < 1e-5

def test_my_pow_negative_base():
    assert abs(my_pow(-2.0, 2) - 4.0) < 1e-5
