"""Tests for Distinct Subsequences kata."""

def test_num_distinct_example1():
    from template import num_distinct
    assert num_distinct("rabbbit", "rabbit") == 3

def test_num_distinct_example2():
    from template import num_distinct
    assert num_distinct("babgbag", "bag") == 5

def test_num_distinct_empty():
    from template import num_distinct
    assert num_distinct("abc", "") == 1

def test_num_distinct_no_match():
    from template import num_distinct
    assert num_distinct("abc", "def") == 0
