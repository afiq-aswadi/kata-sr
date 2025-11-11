"""Tests for Distinct Subsequences kata."""

try:
    from user_kata import num_distinct
except ImportError:
    from .reference import num_distinct


def test_num_distinct_example1():
    assert num_distinct("rabbbit", "rabbit") == 3

def test_num_distinct_example2():
    assert num_distinct("babgbag", "bag") == 5

def test_num_distinct_empty():
    assert num_distinct("abc", "") == 1

def test_num_distinct_no_match():
    assert num_distinct("abc", "def") == 0
