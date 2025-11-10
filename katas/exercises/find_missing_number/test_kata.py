"""Tests for find missing number kata."""

import pytest


def test_missing_number_sum_basic():
    from template import missing_number_sum

    result = missing_number_sum([3, 0, 1])
    assert result == 2


def test_missing_number_sum_zero():
    from template import missing_number_sum

    result = missing_number_sum([1, 2, 3])
    assert result == 0


def test_missing_number_sum_last():
    from template import missing_number_sum

    result = missing_number_sum([0, 1, 2])
    assert result == 3


def test_missing_number_sum_larger():
    from template import missing_number_sum

    result = missing_number_sum([9, 6, 4, 2, 3, 5, 7, 0, 1])
    assert result == 8


def test_missing_number_sum_single():
    from template import missing_number_sum

    result = missing_number_sum([0])
    assert result == 1


def test_missing_number_xor_basic():
    from template import missing_number_xor

    result = missing_number_xor([3, 0, 1])
    assert result == 2


def test_missing_number_xor_zero():
    from template import missing_number_xor

    result = missing_number_xor([1, 2, 3])
    assert result == 0


def test_missing_number_xor_last():
    from template import missing_number_xor

    result = missing_number_xor([0, 1, 2])
    assert result == 3


def test_missing_number_xor_larger():
    from template import missing_number_xor

    result = missing_number_xor([9, 6, 4, 2, 3, 5, 7, 0, 1])
    assert result == 8


def test_missing_number_xor_single():
    from template import missing_number_xor

    result = missing_number_xor([0])
    assert result == 1
