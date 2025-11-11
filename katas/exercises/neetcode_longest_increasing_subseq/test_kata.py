"""Tests for Longest Increasing Subsequence kata."""

try:
    from user_kata import length_of_lis
except ImportError:
    from .reference import length_of_lis


def test_length_of_lis_example1():
    assert length_of_lis([10,9,2,5,3,7,101,18]) == 4

def test_length_of_lis_example2():
    assert length_of_lis([0,1,0,3,2,3]) == 4

def test_length_of_lis_example3():
    assert length_of_lis([7,7,7,7,7,7,7]) == 1

def test_length_of_lis_single():
    assert length_of_lis([1]) == 1

def test_length_of_lis_ascending():
    assert length_of_lis([1,2,3,4,5]) == 5
