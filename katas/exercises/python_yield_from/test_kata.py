"""
Tests for Python Yield From kata
"""

import pytest

try:
    from user_kata import flatten_nested
except ImportError:
    from .reference import flatten_nested


def test_flatten_nested_basic():
    """Test flatten_nested with basic nested structure"""

    nested = [1, [2, 3], [4, [5, 6]], 7]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5, 6, 7]


def test_flatten_nested_deeply_nested():
    """Test flatten_nested with deeply nested structure"""

    nested = [1, [2, [3, [4, [5]]]]]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_empty():
    """Test flatten_nested with empty lists"""

    nested = [[], [1, []], [[]], [2]]
    result = list(flatten_nested(nested))

    assert result == [1, 2]


def test_flatten_nested_all_flat():
    """Test flatten_nested with already flat list"""

    nested = [1, 2, 3, 4, 5]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_all_nested():
    """Test flatten_nested with fully nested structure"""

    nested = [[[[[1]]]]]
    result = list(flatten_nested(nested))

    assert result == [1]


def test_flatten_nested_empty_list():
    """Test flatten_nested with completely empty list"""

    result = list(flatten_nested([]))
    assert result == []


def test_flatten_nested_mixed_depths():
    """Test flatten_nested with mixed nesting depths"""

    nested = [1, [2], [[3]], [[[4]]], 5]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_preserves_order():
    """Test that flatten_nested preserves order"""

    nested = [5, [4, [3]], 2, [1]]
    result = list(flatten_nested(nested))

    assert result == [5, 4, 3, 2, 1]


def test_flatten_nested_strings():
    """Test flatten_nested with strings"""

    nested = ["a", ["b", "c"], [["d"]]]
    result = list(flatten_nested(nested))

    assert result == ["a", "b", "c", "d"]


def test_flatten_nested_mixed_types():
    """Test flatten_nested with mixed types"""

    nested = [1, ["two", [3.0, [None, True]]]]
    result = list(flatten_nested(nested))

    assert result == [1, "two", 3.0, None, True]


def test_flatten_nested_large_structure():
    """Test flatten_nested with larger structure"""

    nested = [
        1,
        [2, 3],
        [4, [5, 6, [7, 8]]],
        9,
        [10, [11, [12]]]
    ]

    result = list(flatten_nested(nested))
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    assert result == expected


def test_flatten_nested_only_nested_lists():
    """Test flatten_nested with only nested lists"""

    nested = [[1], [2], [3]]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3]


def test_returns_generator():
    """Test that function returns a generator"""

    result = flatten_nested([1, [2, 3]])

    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


def test_generator_is_lazy():
    """Test that generator doesn't process until consumed"""

    nested = [1, [2, [3, [4, [5]]]]]
    gen = flatten_nested(nested)

    # Generator shouldn't process until we consume it
    first = next(gen)
    assert first == 1

    second = next(gen)
    assert second == 2


def test_complex_nesting_pattern():
    """Test with complex nesting pattern"""

    nested = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5, 6, 7, 8]


def test_empty_nested_lists():
    """Test with multiple empty nested lists"""

    nested = [[], [[]], [[[]]], 1, []]
    result = list(flatten_nested(nested))

    assert result == [1]


def test_single_element():
    """Test with single element at various nesting levels"""

    assert list(flatten_nested([1])) == [1]
    assert list(flatten_nested([[1]])) == [1]
    assert list(flatten_nested([[[1]]])) == [1]
