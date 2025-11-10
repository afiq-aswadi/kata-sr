"""Tests for Palindrome Partitioning kata."""

def test_partition_example1():
    from template import partition
    result = partition("aab")
    result = [sorted(part) for part in result]
    result = sorted(result)
    expected = [["a","a","b"],["aa","b"]]
    expected = [sorted(part) for part in expected]
    expected = sorted(expected)
    assert result == expected

def test_partition_example2():
    from template import partition
    assert partition("a") == [["a"]]

def test_partition_all_same():
    from template import partition
    result = partition("aaa")
    result = sorted([sorted(part) for part in result])
    expected = [["a","a","a"],["a","aa"],["aa","a"],["aaa"]]
    expected = sorted([sorted(part) for part in expected])
    assert result == expected

def test_partition_no_palindrome():
    from template import partition
    result = partition("ab")
    result = [sorted(part) for part in result]
    result = sorted(result)
    expected = [["a","b"]]
    expected = [sorted(part) for part in expected]
    expected = sorted(expected)
    assert result == expected
