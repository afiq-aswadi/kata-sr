"""Tests for Partition Labels kata."""

def test_partition_labels_example1():
    from template import partition_labels
    assert partition_labels("ababcbacadefegdehijhklij") == [9,7,8]

def test_partition_labels_example2():
    from template import partition_labels
    assert partition_labels("eccbbbbdec") == [10]

def test_partition_labels_single():
    from template import partition_labels
    assert partition_labels("a") == [1]

def test_partition_labels_all_different():
    from template import partition_labels
    assert partition_labels("abcd") == [1,1,1,1]

def test_partition_labels_repeated():
    from template import partition_labels
    assert partition_labels("aaa") == [3]
