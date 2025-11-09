"""
Tests for Python Generators kata

Tests cover:
- Basic generator functionality (yield)
- Generator pipelines
- Bidirectional communication with send()
- Exception handling with throw()
- Cleanup with try/finally
- yield from delegation
- Memory efficiency
- Integration with itertools
"""

import pytest
import sys
import itertools
from typing import List, Iterator, Any


def test_batch_generator_basic():
    """Test batch_generator yields correct batches"""
    from template import batch_generator

    data = list(range(10))
    batches = list(batch_generator(data, 3))

    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]


def test_batch_generator_exact_division():
    """Test batch_generator with exact division"""
    from template import batch_generator

    data = list(range(12))
    batches = list(batch_generator(data, 4))

    assert len(batches) == 3
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6, 7]
    assert batches[2] == [8, 9, 10, 11]


def test_batch_generator_empty():
    """Test batch_generator with empty sequence"""
    from template import batch_generator

    batches = list(batch_generator([], 3))
    assert batches == []


def test_batch_generator_single_batch():
    """Test batch_generator when batch size is larger than data"""
    from template import batch_generator

    data = [1, 2, 3]
    batches = list(batch_generator(data, 10))

    assert len(batches) == 1
    assert batches[0] == [1, 2, 3]


def test_generator_pipeline():
    """Test generator pipeline chains multiple generators"""
    from template import generator_pipeline

    # Pipeline: numbers -> square -> filter even -> double
    numbers = range(10)
    result = list(generator_pipeline(numbers))

    # Expected: 0^2=0(even)->0, 1^2=1(odd), 2^2=4(even)->8, 3^2=9(odd),
    #           4^2=16(even)->32, 5^2=25(odd), 6^2=36(even)->72, 7^2=49(odd),
    #           8^2=64(even)->128, 9^2=81(odd)
    expected = [0, 8, 32, 72, 128]
    assert result == expected


def test_running_average_basic():
    """Test running_average computes correct averages"""
    from template import running_average

    avg = running_average()
    next(avg)  # Prime the generator

    assert avg.send(10) == 10.0
    assert avg.send(20) == 15.0
    assert avg.send(30) == 20.0


def test_running_average_negative_numbers():
    """Test running_average with negative numbers"""
    from template import running_average

    avg = running_average()
    next(avg)  # Prime the generator

    assert avg.send(10) == 10.0
    assert avg.send(-10) == 0.0
    assert avg.send(5) == pytest.approx(5.0 / 3)


def test_running_average_float_precision():
    """Test running_average maintains precision"""
    from template import running_average

    avg = running_average()
    next(avg)  # Prime the generator

    assert avg.send(1) == 1.0
    assert avg.send(2) == 1.5
    assert avg.send(3) == 2.0
    assert avg.send(4) == 2.5


def test_file_reader_generator_cleanup():
    """Test file_reader_generator executes cleanup code"""
    from template import file_reader_generator
    import tempfile
    import os

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        # Read first two lines
        gen = file_reader_generator(temp_path)
        lines = [next(gen).strip(), next(gen).strip()]

        assert lines == ["line1", "line2"]

        # Close generator early - should trigger cleanup
        gen.close()

        # Verify cleanup was called (file should be closed)
        # We can't directly test this, but we can consume all lines
        gen2 = file_reader_generator(temp_path)
        all_lines = [line.strip() for line in gen2]
        assert all_lines == ["line1", "line2", "line3"]

    finally:
        os.unlink(temp_path)


def test_file_reader_generator_full_consumption():
    """Test file_reader_generator when fully consumed"""
    from template import file_reader_generator
    import tempfile
    import os

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("a\nb\nc\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        lines = [line.strip() for line in gen]
        assert lines == ["a", "b", "c"]
    finally:
        os.unlink(temp_path)


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


def test_generator_is_lazy():
    """Test that generators are lazy (don't compute until needed)"""
    from template import batch_generator

    call_count = [0]

    def counting_generator():
        for i in range(1000000):
            call_count[0] += 1
            yield i

    # Create generator but don't consume
    gen = batch_generator(counting_generator(), 10)

    # Should not have executed anything yet
    # (Note: batch_generator might consume one batch, which is fine)
    # The key is it shouldn't consume ALL items
    assert call_count[0] < 100  # Should be way less than 1000000

    # Consume first batch
    first_batch = next(gen)
    assert len(first_batch) == 10

    # Should have consumed only what's needed for first batch
    assert call_count[0] <= 20  # Some buffer is acceptable


def test_memory_efficiency():
    """Test that generators don't materialize entire sequence"""
    from template import batch_generator

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


def test_send_protocol():
    """Test that send() works correctly"""
    from template import running_average

    avg = running_average()

    # First next() to prime the generator
    result = next(avg)
    assert result is None  # Initial yield should return None

    # Now use send()
    result1 = avg.send(100)
    assert result1 == 100.0

    result2 = avg.send(200)
    assert result2 == 150.0


def test_generator_expression_vs_list():
    """Test understanding of generator expressions"""
    # This is more of a conceptual test
    # Generator expression
    gen_expr = (x**2 for x in range(5))
    assert hasattr(gen_expr, '__iter__')
    assert hasattr(gen_expr, '__next__')

    # List comprehension
    list_comp = [x**2 for x in range(5)]
    assert isinstance(list_comp, list)

    # Generator can only be consumed once
    result1 = list(gen_expr)
    result2 = list(gen_expr)  # Should be empty
    assert result1 == [0, 1, 4, 9, 16]
    assert result2 == []


def test_stopiteration_handling():
    """Test that StopIteration is raised correctly"""
    from template import batch_generator

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
    """Test that generators work with itertools"""
    from template import batch_generator

    data = range(20)
    gen = batch_generator(data, 5)

    # Use itertools.islice to get first 2 batches
    first_two = list(itertools.islice(gen, 2))
    assert len(first_two) == 2
    assert first_two[0] == [0, 1, 2, 3, 4]
    assert first_two[1] == [5, 6, 7, 8, 9]

    # Use itertools.chain to flatten
    from template import batch_generator
    gen2 = batch_generator(data, 3)
    flattened = list(itertools.chain.from_iterable(gen2))
    assert flattened == list(range(20))


def test_yield_from_delegation():
    """Test that flatten_nested uses yield from correctly"""
    from template import flatten_nested

    # Complex nested structure
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


def test_generator_state_preservation():
    """Test that generator state is preserved between yields"""
    from template import running_average

    avg = running_average()
    next(avg)

    # Send values and verify state is maintained
    values = [5, 10, 15, 20, 25]
    results = [avg.send(v) for v in values]

    # Running averages: 5, 7.5, 10, 12.5, 15
    expected = [5.0, 7.5, 10.0, 12.5, 15.0]

    assert results == expected


def test_generator_pipeline_memory_efficiency():
    """Test that pipeline doesn't materialize intermediate results"""
    from template import generator_pipeline

    # Large input - pipeline should handle this efficiently
    large_input = range(10**5)
    gen = generator_pipeline(large_input)

    # Get first 10 results
    first_10 = list(itertools.islice(gen, 10))

    # Verify first few results
    assert first_10[0] == 0  # 0^2=0(even)->0
    assert first_10[1] == 8  # 2^2=4(even)->8


def test_file_reader_exception_cleanup():
    """Test that cleanup happens even with exceptions"""
    from template import file_reader_generator
    import tempfile
    import os

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        gen = file_reader_generator(temp_path)
        next(gen)  # Read first line

        # Force exception by closing generator
        try:
            gen.throw(RuntimeError("test error"))
        except RuntimeError:
            pass  # Expected

        # Generator should have cleaned up
        # We can verify by creating a new generator and reading the file
        gen2 = file_reader_generator(temp_path)
        lines = list(gen2)
        assert len(lines) == 3

    finally:
        os.unlink(temp_path)
