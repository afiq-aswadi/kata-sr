"""
Tests for Python Generator Send kata
"""

import pytest


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


def test_multiple_generators():
    """Test that multiple generator instances maintain separate state"""
    from template import running_average

    avg1 = running_average()
    avg2 = running_average()

    next(avg1)
    next(avg2)

    # Send to first generator
    assert avg1.send(10) == 10.0
    assert avg1.send(20) == 15.0

    # Send to second generator (should have separate state)
    assert avg2.send(100) == 100.0
    assert avg2.send(200) == 150.0

    # First generator should still have its own state
    assert avg1.send(30) == 20.0


def test_large_sequence():
    """Test running_average with larger sequence"""
    from template import running_average

    avg = running_average()
    next(avg)

    # Sum of 1..10 = 55, average = 5.5
    for i in range(1, 11):
        result = avg.send(i)

    assert result == 5.5


def test_floating_point_values():
    """Test running_average with floating point values"""
    from template import running_average

    avg = running_average()
    next(avg)

    assert avg.send(1.5) == 1.5
    assert avg.send(2.5) == 2.0
    assert avg.send(3.0) == pytest.approx(7.0 / 3)


def test_zeros():
    """Test running_average with zeros"""
    from template import running_average

    avg = running_average()
    next(avg)

    assert avg.send(0) == 0.0
    assert avg.send(0) == 0.0
    assert avg.send(10) == pytest.approx(10.0 / 3)


def test_prime_before_send():
    """Test that generator must be primed before send()"""
    from template import running_average

    avg = running_average()

    # Without priming, send() should raise TypeError
    # (can't send to a just-started generator)
    with pytest.raises(TypeError):
        avg.send(10)


def test_alternating_positive_negative():
    """Test with alternating positive and negative values"""
    from template import running_average

    avg = running_average()
    next(avg)

    assert avg.send(10) == 10.0
    assert avg.send(-5) == 2.5
    assert avg.send(10) == 5.0
    assert avg.send(-5) == 2.5


def test_is_infinite_generator():
    """Test that generator continues indefinitely"""
    from template import running_average

    avg = running_average()
    next(avg)

    # Send 100 values - generator should handle this
    for i in range(1, 101):
        result = avg.send(i)

    # Average of 1..100 is 50.5
    assert result == 50.5


def test_returns_generator():
    """Test that function returns a generator"""
    from template import running_average

    gen = running_average()

    assert hasattr(gen, '__iter__')
    assert hasattr(gen, '__next__')
    assert hasattr(gen, 'send')
