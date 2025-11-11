"""Tests for find missing number kata."""

import pytest


try:
    from user_kata import missing_number_sum
    from user_kata import missing_number_xor
except ImportError:
    from .reference import missing_number_sum
    from .reference import missing_number_xor


def test_missing_number_sum_basic():

    result = missing_number_sum([3, 0, 1])
    assert result == 2


def test_missing_number_sum_zero():

    result = missing_number_sum([1, 2, 3])
    assert result == 0


def test_missing_number_sum_last():

    result = missing_number_sum([0, 1, 2])
    assert result == 3


def test_missing_number_sum_larger():

    result = missing_number_sum([9, 6, 4, 2, 3, 5, 7, 0, 1])
    assert result == 8


def test_missing_number_sum_single():

    result = missing_number_sum([0])
    assert result == 1


def test_missing_number_xor_basic():

    result = missing_number_xor([3, 0, 1])
    assert result == 2


def test_missing_number_xor_zero():

    result = missing_number_xor([1, 2, 3])
    assert result == 0


def test_missing_number_xor_last():

    result = missing_number_xor([0, 1, 2])
    assert result == 3


def test_missing_number_xor_larger():

    result = missing_number_xor([9, 6, 4, 2, 3, 5, 7, 0, 1])
    assert result == 8


def test_missing_number_xor_single():

    result = missing_number_xor([0])
    assert result == 1
