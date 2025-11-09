"""
Tests for Python Generator Pipeline kata
"""

import pytest
import itertools


def test_generator_pipeline_basic():
    """Test generator_pipeline with basic input"""
    from template import generator_pipeline

    result = list(generator_pipeline(range(5)))
    # 0^2=0(even)->0, 2^2=4(even)->8, 4^2=16(even)->32
    expected = [0, 8, 32]
    assert result == expected


def test_generator_pipeline_larger_range():
    """Test generator_pipeline with larger range"""
    from template import generator_pipeline

    result = list(generator_pipeline(range(10)))
    # Even squares doubled: 0->0, 4->8, 16->32, 36->72, 64->128
    expected = [0, 8, 32, 72, 128]
    assert result == expected


def test_generator_pipeline_empty():
    """Test generator_pipeline with empty input"""
    from template import generator_pipeline

    result = list(generator_pipeline([]))
    assert result == []


def test_generator_pipeline_single_element():
    """Test generator_pipeline with single element"""
    from template import generator_pipeline

    # 2^2 = 4 (even) -> 8
    result = list(generator_pipeline([2]))
    assert result == [8]

    # 3^2 = 9 (odd) -> filtered out
    result = list(generator_pipeline([3]))
    assert result == []


def test_generator_pipeline_all_odd():
    """Test generator_pipeline filters out all odd squares"""
    from template import generator_pipeline

    # 1^2=1, 3^2=9, 5^2=25, 7^2=49 - all odd
    result = list(generator_pipeline([1, 3, 5, 7]))
    assert result == []


def test_generator_pipeline_all_even_input():
    """Test generator_pipeline with all even input"""
    from template import generator_pipeline

    # 0^2=0, 2^2=4, 4^2=16, 6^2=36
    result = list(generator_pipeline([0, 2, 4, 6]))
    expected = [0, 8, 32, 72]
    assert result == expected


def test_pipeline_is_lazy():
    """Test that pipeline doesn't process until needed"""
    from template import generator_pipeline

    call_count = [0]

    def counting_generator():
        for i in range(10**5):
            call_count[0] += 1
            yield i

    # Create pipeline but don't consume
    gen = generator_pipeline(counting_generator())

    # Should not have processed much yet
    assert call_count[0] < 1000

    # Get first result
    first = next(gen)
    assert first == 0

    # Should have processed only what's needed
    assert call_count[0] < 1000


def test_memory_efficiency():
    """Test that pipeline handles large input without memory issues"""
    from template import generator_pipeline

    # Large range - would be huge if materialized
    large_range = range(10**6)
    gen = generator_pipeline(large_range)

    # Get first few results
    first_five = list(itertools.islice(gen, 5))
    assert first_five == [0, 8, 32, 72, 128]


def test_pipeline_with_itertools():
    """Test that pipeline works with itertools"""
    from template import generator_pipeline

    gen = generator_pipeline(range(20))

    # Use itertools.islice to get first 3 results
    first_three = list(itertools.islice(gen, 3))
    assert first_three == [0, 8, 32]


def test_generator_is_not_reusable():
    """Test that generator can only be consumed once"""
    from template import generator_pipeline

    gen = generator_pipeline(range(5))

    # First consumption
    result1 = list(gen)
    assert result1 == [0, 8, 32]

    # Second consumption should be empty
    result2 = list(gen)
    assert result2 == []


def test_negative_numbers():
    """Test generator_pipeline with negative numbers"""
    from template import generator_pipeline

    # -2^2=4(even)->8, -1^2=1(odd), 0^2=0(even)->0, 1^2=1(odd), 2^2=4(even)->8
    result = list(generator_pipeline([-2, -1, 0, 1, 2]))
    expected = [8, 0, 8]
    assert result == expected


def test_returns_generator():
    """Test that function returns a generator"""
    from template import generator_pipeline

    result = generator_pipeline(range(5))

    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


def test_intermediate_stages_not_materialized():
    """Test that intermediate stages aren't materialized as lists"""
    from template import generator_pipeline

    # This is more of a conceptual test
    # The pipeline should pass items through without creating intermediate lists
    gen = generator_pipeline(range(100))

    # Take just first item
    first = next(gen)
    assert first == 0

    # Should be able to continue
    second = next(gen)
    assert second == 8
