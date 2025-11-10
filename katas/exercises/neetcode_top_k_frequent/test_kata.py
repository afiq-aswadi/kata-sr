"""Tests for Top K Frequent Elements kata."""

def test_top_k_frequent_example1():
    from template import top_k_frequent
    result = top_k_frequent([1,1,1,2,2,3], 2)
    assert sorted(result) == [1, 2]

def test_top_k_frequent_example2():
    from template import top_k_frequent
    assert top_k_frequent([1], 1) == [1]

def test_top_k_frequent_all():
    from template import top_k_frequent
    result = top_k_frequent([4,1,-1,2,-1,2,3], 2)
    assert sorted(result) == [-1, 2]
