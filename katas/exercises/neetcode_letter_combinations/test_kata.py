"""Tests for Letter Combinations kata."""

try:
    from user_kata import letter_combinations
except ImportError:
    from .reference import letter_combinations


def test_letter_combinations_example1():
    result = sorted(letter_combinations("23"))
    expected = sorted(["ad","ae","af","bd","be","bf","cd","ce","cf"])
    assert result == expected

def test_letter_combinations_example2():
    assert letter_combinations("") == []

def test_letter_combinations_example3():
    result = sorted(letter_combinations("2"))
    expected = sorted(["a","b","c"])
    assert result == expected

def test_letter_combinations_single_digit():
    result = sorted(letter_combinations("7"))
    expected = sorted(["p","q","r","s"])
    assert result == expected
