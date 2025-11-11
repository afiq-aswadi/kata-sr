"""Tests for Partition Equal Subset Sum kata."""

try:
    from user_kata import can_partition
except ImportError:
    from .reference import can_partition


def test_can_partition_example1():
    assert can_partition([1,5,11,5]) == True

def test_can_partition_example2():
    assert can_partition([1,2,3,5]) == False

def test_can_partition_single():
    assert can_partition([1]) == False

def test_can_partition_two_equal():
    assert can_partition([1,1]) == True

def test_can_partition_odd_sum():
    assert can_partition([1,2,3]) == False
