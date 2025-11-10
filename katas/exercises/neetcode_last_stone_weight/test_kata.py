"""Tests for Last Stone Weight kata."""

def test_last_stone_weight_example():
    from template import last_stone_weight
    assert last_stone_weight([2,7,4,1,8,1]) == 1

def test_last_stone_weight_all_equal():
    from template import last_stone_weight
    assert last_stone_weight([1]) == 1

def test_last_stone_weight_two_stones():
    from template import last_stone_weight
    assert last_stone_weight([3,7]) == 4

def test_last_stone_weight_all_smashed():
    from template import last_stone_weight
    assert last_stone_weight([2,2]) == 0
