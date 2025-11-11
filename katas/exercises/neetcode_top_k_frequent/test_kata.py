"""Tests for Top K Frequent Elements kata."""

try:
    from user_kata import top_k_frequent
except ImportError:
    from .reference import top_k_frequent


def test_top_k_frequent_example1():
    result = top_k_frequent([1,1,1,2,2,3], 2)
    assert sorted(result) == [1, 2]

def test_top_k_frequent_example2():
    assert top_k_frequent([1], 1) == [1]

def test_top_k_frequent_all():
    result = top_k_frequent([4,1,-1,2,-1,2,3], 2)
    assert sorted(result) == [-1, 2]
