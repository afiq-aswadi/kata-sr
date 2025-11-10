"""Tests for Partition Equal Subset Sum kata."""

def test_can_partition_example1():
    from template import can_partition
    assert can_partition([1,5,11,5]) == True

def test_can_partition_example2():
    from template import can_partition
    assert can_partition([1,2,3,5]) == False

def test_can_partition_single():
    from template import can_partition
    assert can_partition([1]) == False

def test_can_partition_two_equal():
    from template import can_partition
    assert can_partition([1,1]) == True

def test_can_partition_odd_sum():
    from template import can_partition
    assert can_partition([1,2,3]) == False
