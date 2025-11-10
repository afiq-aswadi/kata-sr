"""Tests for remove duplicates kata."""

import pytest


def test_remove_duplicates_basic():
    from template import remove_duplicates

    nums = [1, 1, 2]
    length = remove_duplicates(nums)
    assert length == 2
    assert nums[:length] == [1, 2]


def test_remove_duplicates_multiple():
    from template import remove_duplicates

    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    length = remove_duplicates(nums)
    assert length == 5
    assert nums[:length] == [0, 1, 2, 3, 4]


def test_remove_duplicates_no_duplicates():
    from template import remove_duplicates

    nums = [1, 2, 3, 4]
    length = remove_duplicates(nums)
    assert length == 4
    assert nums[:length] == [1, 2, 3, 4]


def test_remove_duplicates_all_same():
    from template import remove_duplicates

    nums = [1, 1, 1, 1]
    length = remove_duplicates(nums)
    assert length == 1
    assert nums[:length] == [1]


def test_remove_duplicates_empty():
    from template import remove_duplicates

    nums = []
    length = remove_duplicates(nums)
    assert length == 0


def test_remove_duplicates_allow_twice_basic():
    from template import remove_duplicates_allow_twice

    nums = [1, 1, 1, 2, 2, 3]
    length = remove_duplicates_allow_twice(nums)
    assert length == 5
    assert nums[:length] == [1, 1, 2, 2, 3]


def test_remove_duplicates_allow_twice_multiple():
    from template import remove_duplicates_allow_twice

    nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    length = remove_duplicates_allow_twice(nums)
    assert length == 7
    assert nums[:length] == [0, 0, 1, 1, 2, 3, 3]
