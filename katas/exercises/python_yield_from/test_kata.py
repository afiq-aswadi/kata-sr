"""
Tests for Python Yield From kata
"""

import pytest


def test_flatten_nested_basic():
    """Test flatten_nested with basic nested structure"""
    from template import flatten_nested

    nested = [1, [2, 3], [4, [5, 6]], 7]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5, 6, 7]


def test_flatten_nested_deeply_nested():
    """Test flatten_nested with deeply nested structure"""
    from template import flatten_nested

    nested = [1, [2, [3, [4, [5]]]]]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_empty():
    """Test flatten_nested with empty lists"""
    from template import flatten_nested

    nested = [[], [1, []], [[]], [2]]
    result = list(flatten_nested(nested))

    assert result == [1, 2]


def test_flatten_nested_all_flat():
    """Test flatten_nested with already flat list"""
    from template import flatten_nested

    nested = [1, 2, 3, 4, 5]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_all_nested():
    """Test flatten_nested with fully nested structure"""
    from template import flatten_nested

    nested = [[[[[1]]]]]
    result = list(flatten_nested(nested))

    assert result == [1]


def test_flatten_nested_empty_list():
    """Test flatten_nested with completely empty list"""
    from template import flatten_nested

    result = list(flatten_nested([]))
    assert result == []


def test_flatten_nested_mixed_depths():
    """Test flatten_nested with mixed nesting depths"""
    from template import flatten_nested

    nested = [1, [2], [[3]], [[[4]]], 5]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5]


def test_flatten_nested_preserves_order():
    """Test that flatten_nested preserves order"""
    from template import flatten_nested

    nested = [5, [4, [3]], 2, [1]]
    result = list(flatten_nested(nested))

    assert result == [5, 4, 3, 2, 1]


def test_flatten_nested_strings():
    """Test flatten_nested with strings"""
    from template import flatten_nested

    nested = ["a", ["b", "c"], [["d"]]]
    result = list(flatten_nested(nested))

    assert result == ["a", "b", "c", "d"]


def test_flatten_nested_mixed_types():
    """Test flatten_nested with mixed types"""
    from template import flatten_nested

    nested = [1, ["two", [3.0, [None, True]]]]
    result = list(flatten_nested(nested))

    assert result == [1, "two", 3.0, None, True]


def test_flatten_nested_large_structure():
    """Test flatten_nested with larger structure"""
    from template import flatten_nested

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
    from template import flatten_nested

    nested = [[1], [2], [3]]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3]


def test_returns_generator():
    """Test that function returns a generator"""
    from template import flatten_nested

    result = flatten_nested([1, [2, 3]])

    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


def test_generator_is_lazy():
    """Test that generator doesn't process until consumed"""
    from template import flatten_nested

    nested = [1, [2, [3, [4, [5]]]]]
    gen = flatten_nested(nested)

    # Generator shouldn't process until we consume it
    first = next(gen)
    assert first == 1

    second = next(gen)
    assert second == 2


def test_complex_nesting_pattern():
    """Test with complex nesting pattern"""
    from template import flatten_nested

    nested = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
    result = list(flatten_nested(nested))

    assert result == [1, 2, 3, 4, 5, 6, 7, 8]


def test_empty_nested_lists():
    """Test with multiple empty nested lists"""
    from template import flatten_nested

    nested = [[], [[]], [[[]]], 1, []]
    result = list(flatten_nested(nested))

    assert result == [1]


def test_single_element():
    """Test with single element at various nesting levels"""
    from template import flatten_nested

    assert list(flatten_nested([1])) == [1]
    assert list(flatten_nested([[1]])) == [1]
    assert list(flatten_nested([[[1]]])) == [1]
