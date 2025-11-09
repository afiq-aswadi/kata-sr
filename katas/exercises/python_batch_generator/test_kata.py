"""
Tests for Python Batch Generator kata
"""

import pytest
import itertools

try:
    from user_kata import batch_generator
except ImportError:
    from .reference import batch_generator


def test_batch_generator_basic():
    """Test batch_generator yields correct batches"""

    data = list(range(10))
    batches = list(batch_generator(data, 3))

    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]


def test_batch_generator_exact_division():
    """Test batch_generator with exact division"""

    data = list(range(12))
    batches = list(batch_generator(data, 4))

    assert len(batches) == 3
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6, 7]
    assert batches[2] == [8, 9, 10, 11]


def test_batch_generator_empty():
    """Test batch_generator with empty sequence"""

    batches = list(batch_generator([], 3))
    assert batches == []


def test_batch_generator_single_batch():
    """Test batch_generator when batch size is larger than data"""

    data = [1, 2, 3]
    batches = list(batch_generator(data, 10))

    assert len(batches) == 1
    assert batches[0] == [1, 2, 3]


def test_batch_generator_batch_size_one():
    """Test batch_generator with batch_size=1"""

    data = [1, 2, 3, 4]
    batches = list(batch_generator(data, 1))

    assert len(batches) == 4
    assert batches == [[1], [2], [3], [4]]


def test_generator_is_lazy():
    """Test that generator is lazy (doesn't compute until needed)"""

    call_count = [0]

    def counting_generator():
        for i in range(1000000):
            call_count[0] += 1
            yield i

    # Create generator but don't consume
    gen = batch_generator(counting_generator(), 10)

    # Should not have executed anything yet (or very little)
    assert call_count[0] < 100  # Should be way less than 1000000

    # Consume first batch
    first_batch = next(gen)
    assert len(first_batch) == 10

    # Should have consumed only what's needed
    assert call_count[0] <= 20  # Some buffer is acceptable


def test_memory_efficiency():
    """Test that generator doesn't materialize entire sequence"""

    # Create a large range (would be huge if materialized)
    large_range = range(10**6)

    # Use batch_generator - should handle this fine
    gen = batch_generator(large_range, 100)

    # Get first batch
    first_batch = next(gen)
    assert len(first_batch) == 100
    assert first_batch[0] == 0

    # Get second batch
    second_batch = next(gen)
    assert len(second_batch) == 100
    assert second_batch[0] == 100


def test_stopiteration_handling():
    """Test that StopIteration is raised correctly"""

    data = [1, 2, 3]
    gen = batch_generator(data, 2)

    batch1 = next(gen)
    assert batch1 == [1, 2]

    batch2 = next(gen)
    assert batch2 == [3]

    # Should raise StopIteration
    with pytest.raises(StopIteration):
        next(gen)


def test_itertools_integration():
    """Test that generator works with itertools"""

    data = range(20)
    gen = batch_generator(data, 5)

    # Use itertools.islice to get first 2 batches
    first_two = list(itertools.islice(gen, 2))
    assert len(first_two) == 2
    assert first_two[0] == [0, 1, 2, 3, 4]
    assert first_two[1] == [5, 6, 7, 8, 9]


def test_generator_attributes():
    """Test that function returns a generator"""

    gen = batch_generator(range(10), 3)

    # Should be a generator
    assert hasattr(gen, '__iter__')
    assert hasattr(gen, '__next__')


def test_batch_with_strings():
    """Test batching with different types (strings)"""

    data = "abcdefghij"
    batches = list(batch_generator(data, 3))

    assert len(batches) == 4
    assert batches[0] == ['a', 'b', 'c']
    assert batches[1] == ['d', 'e', 'f']
    assert batches[2] == ['g', 'h', 'i']
    assert batches[3] == ['j']


def test_batch_with_mixed_types():
    """Test batching with mixed types"""

    data = [1, "two", 3.0, None, True]
    batches = list(batch_generator(data, 2))

    assert len(batches) == 3
    assert batches[0] == [1, "two"]
    assert batches[1] == [3.0, None]
    assert batches[2] == [True]
