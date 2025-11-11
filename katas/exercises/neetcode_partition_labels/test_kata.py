"""Tests for Partition Labels kata."""

try:
    from user_kata import partition_labels
except ImportError:
    from .reference import partition_labels


def test_partition_labels_example1():
    assert partition_labels("ababcbacadefegdehijhklij") == [9,7,8]

def test_partition_labels_example2():
    assert partition_labels("eccbbbbdec") == [10]

def test_partition_labels_single():
    assert partition_labels("a") == [1]

def test_partition_labels_all_different():
    assert partition_labels("abcd") == [1,1,1,1]

def test_partition_labels_repeated():
    assert partition_labels("aaa") == [3]
