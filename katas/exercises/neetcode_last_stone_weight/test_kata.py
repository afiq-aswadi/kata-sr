"""Tests for Last Stone Weight kata."""

try:
    from user_kata import last_stone_weight
except ImportError:
    from .reference import last_stone_weight


def test_last_stone_weight_example():
    assert last_stone_weight([2,7,4,1,8,1]) == 1

def test_last_stone_weight_all_equal():
    assert last_stone_weight([1]) == 1

def test_last_stone_weight_two_stones():
    assert last_stone_weight([3,7]) == 4

def test_last_stone_weight_all_smashed():
    assert last_stone_weight([2,2]) == 0
